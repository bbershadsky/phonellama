/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover

import com.google.ai.edge.gallery.rover.benchmark.FrameMetrics
import com.google.ai.edge.gallery.rover.benchmark.LatestSlot
import com.google.ai.edge.gallery.rover.benchmark.dropPercent
import com.google.ai.edge.gallery.rover.benchmark.percentileMs
import com.google.ai.edge.gallery.rover.control.ControlInput
import com.google.ai.edge.gallery.rover.control.ControlLoop
import com.google.ai.edge.gallery.rover.control.FollowController
import com.google.ai.edge.gallery.rover.control.MotionSmoother
import com.google.ai.edge.gallery.rover.control.RawVelocity
import com.google.ai.edge.gallery.rover.inference.Detection
import com.google.ai.edge.gallery.rover.inference.onlyPersons
import com.google.ai.edge.gallery.rover.protocol.CommandGate
import com.google.ai.edge.gallery.rover.protocol.JsonProtocol
import com.google.ai.edge.gallery.rover.protocol.Telemetry
import com.google.ai.edge.gallery.rover.protocol.VelocityCommand
import com.google.ai.edge.gallery.rover.protocol.wheelSpeeds
import com.google.ai.edge.gallery.rover.safety.RoverConfig
import com.google.ai.edge.gallery.rover.safety.SafetyArbiter
import com.google.ai.edge.gallery.rover.safety.StopReason
import com.google.ai.edge.gallery.rover.tracking.SortTracker
import com.google.ai.edge.gallery.rover.tracking.TargetSelector
import com.google.ai.edge.gallery.rover.transport.MockRobotTransport
import com.google.ai.edge.gallery.rover.tracking.TrackStrategy
import com.google.ai.edge.gallery.rover.tracking.firstVisiblePerson
import com.google.ai.edge.gallery.rover.ui.TiltAdvice
import com.google.ai.edge.gallery.rover.ui.contentRect
import com.google.ai.edge.gallery.rover.ui.aimPoint
import com.google.ai.edge.gallery.rover.ui.tiltToCenter
import com.google.ai.edge.gallery.rover.ui.viewPointToNormalized
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RoverLogicTest {
  @Test
  fun commandGateRejectsStaleAndDuplicateSequences() {
    val gate = CommandGate()
    val first = command(seq = 1, timestampMs = 1_000, linear = 0.2f)
    assertTrue(gate.accept(first, nowMs = 1_000))
    assertFalse(gate.accept(first.copy(seq = 1, timestampMs = 1_050), nowMs = 1_050))
    assertFalse(gate.accept(first.copy(seq = 2, timestampMs = 1_000), nowMs = 1_300))
    assertTrue(gate.accept(first.copy(seq = 3, timestampMs = 1_300), nowMs = 1_300))
  }

  @Test
  fun mockTransportStopsWhenTtlExpires() = runBlocking {
    val transport = MockRobotTransport()
    transport.connect()
    val move = command(seq = 1, timestampMs = 1_000, linear = 0.2f)
    assertTrue(transport.offer(move, nowMs = 1_000))
    assertFalse(transport.offer(move.copy(seq = 2, timestampMs = 1_000), nowMs = 1_250))
    assertEquals(0.2f, transport.effective(1_100)!!.linearMps, 0.001f)
    assertEquals(0f, transport.effective(1_250)!!.linearMps, 0.001f)
    assertEquals(0f, transport.effective(1_250)!!.angularRps, 0.001f)
    assertEquals(1, transport.rejectedCount)
  }

  @Test
  fun disconnectedTransportRejectsCommands() {
    val transport = MockRobotTransport()
    assertFalse(transport.offer(command(seq = 1, timestampMs = 0, linear = 0.1f), nowMs = 0))
    assertNull(transport.effective(0))
  }

  @Test
  fun velocityJsonRoundTripKeepsWireNames() {
    val command = command(seq = 1842, timestampMs = 1_727_270_000_123, linear = 0.25f, angular = -0.125f)
    val text = JsonProtocol.encode(command)
    assertTrue(text.contains("\"linear_mps\""))
    assertTrue(text.contains("\"angular_rps\""))
    assertTrue(text.contains("\"ttl_ms\""))
    val decoded = JsonProtocol.decodeCommand(text)
    assertEquals(command, decoded)
    val telemetry = Telemetry(seq = 4, frontDistanceMm = 820, batteryV = 12f)
    assertEquals(telemetry, JsonProtocol.decodeTelemetry(JsonProtocol.encode(telemetry)))
  }

  @Test
  fun differentialDriveMatchesSpec() {
    val straight = wheelSpeeds(linearMps = 0.2f, angularRps = 0f, wheelSeparationM = 0.3f)
    assertEquals(0.2f, straight.first, 0.001f)
    assertEquals(0.2f, straight.second, 0.001f)
    val spin = wheelSpeeds(linearMps = 0f, angularRps = 1f, wheelSeparationM = 0.2f)
    assertEquals(-0.1f, spin.first, 0.001f)
    assertEquals(0.1f, spin.second, 0.001f)
  }

  @Test
  fun followControllerTurnsTowardTargetAndKeepsDeadband() {
    val controller = FollowController()
    val rightAndFar =
      controller.compute(track(id = 1, left = 0.7f, top = 0.2f, right = 0.9f, bottom = 0.8f))
    assertTrue(rightAndFar.angularRps < 0f)
    assertTrue(rightAndFar.linearMps > 0f)
    val centered =
      controller.compute(track(id = 1, left = 0.325f, top = 0.1f, right = 0.675f, bottom = 0.9f))
    assertEquals(0f, centered.linearMps, 0.001f)
    assertEquals(0f, centered.angularRps, 0.001f)
  }

  @Test
  fun smootherLimitsAccelerationAndResetStops() {
    val smoother = MotionSmoother(RoverConfig(maxLinearAccelMps2 = 0.5f, maxAngularAccelRps2 = 1f))
    assertEquals(0f, smoother.smooth(RawVelocity(1f, 1f), nowMs = 0).linearMps, 0.001f)
    val stepped = smoother.smooth(RawVelocity(1f, 1f), nowMs = 100)
    assertEquals(0.05f, stepped.linearMps, 0.001f)
    assertEquals(0.1f, stepped.angularRps, 0.001f)
    smoother.reset()
    assertEquals(0f, smoother.smooth(RawVelocity(1f, 0f), nowMs = 500).linearMps, 0.001f)
  }

  @Test
  fun safetyStopOverridesALockedTarget() {
    val safety = SafetyArbiter()
    val person = track(id = 3, left = 0.3f, top = 0.1f, right = 0.6f, bottom = 0.9f, score = 0.9f)
    assertEquals(
      StopReason.ESTOP,
      safety.evaluate(person, hasLock = true, Telemetry(estop = true, frontDistanceMm = 2000), RoverConfig()),
    )
    assertEquals(
      StopReason.OBSTACLE,
      safety.evaluate(person, hasLock = true, Telemetry(frontDistanceMm = 200), RoverConfig()),
    )
    assertEquals(
      StopReason.SENSOR_FAULT,
      safety.evaluate(person, hasLock = true, Telemetry(fault = true, frontDistanceMm = 2000), RoverConfig()),
    )
    assertEquals(
      StopReason.LOW_BATTERY,
      safety.evaluate(person, hasLock = true, Telemetry(batteryV = 9f, frontDistanceMm = 2000), RoverConfig()),
    )
    assertEquals(
      StopReason.LOW_CONFIDENCE,
      safety.evaluate(person.copy(score = 0.2f), hasLock = true, Telemetry(frontDistanceMm = 2000), RoverConfig()),
    )
    assertEquals(
      StopReason.NO_TARGET,
      safety.evaluate(null, hasLock = false, Telemetry(frontDistanceMm = 2000), RoverConfig()),
    )
  }

  @Test
  fun lockedTargetSurvivesAMissAndDoesNotSwitchPeople() {
    val loop = ControlLoop()
    val left = detection(0.10f, 0.15f, 0.40f, 0.90f)
    val right = detection(0.60f, 0.15f, 0.90f, 0.90f)
    val seen = loop.step(input(listOf(left, right), locked = null, nowMs = 0))
    val leftId = seen.tracks.minBy { it.centerX }.id
    val rightId = seen.tracks.maxBy { it.centerX }.id
    assertNotEquals(leftId, rightId)

    val missed = loop.step(input(listOf(right), locked = leftId, nowMs = 50))
    val lost = missed.tracks.first { it.id == leftId }
    assertEquals(1, lost.lostFrames)
    assertEquals(leftId, missed.lockedTrackId)
    assertEquals(StopReason.LOST_TARGET, missed.reason)
    assertEquals(0f, missed.command.linearMps, 0.001f)
    assertEquals(0f, missed.command.angularRps, 0.001f)
    assertEquals(rightId, missed.tracks.first { it.lostFrames == 0 }.id)

    val returned = loop.step(input(listOf(left, right), locked = leftId, nowMs = 100))
    assertEquals(0, returned.tracks.first { it.id == leftId }.lostFrames)
    assertEquals(leftId, returned.lockedTrackId)
  }

  @Test
  fun obstacleZeroesVelocityEvenAfterTheRobotWasMoving() {
    val loop = ControlLoop()
    val person = detection(0.40f, 0.10f, 0.55f, 0.90f)
    val first = loop.step(input(listOf(person), locked = null, nowMs = 0))
    val id = first.tracks.single().id
    loop.step(input(listOf(person), locked = id, nowMs = 0))
    val moving = loop.step(input(listOf(person), locked = id, nowMs = 100))
    assertEquals(StopReason.NONE, moving.reason)
    assertTrue(moving.command.linearMps > 0f)
    val blocked =
      loop.step(
        input(
          listOf(person),
          locked = id,
          nowMs = 200,
          telemetry = Telemetry(frontDistanceMm = 150, estop = false),
        )
      )
    assertEquals(StopReason.OBSTACLE, blocked.reason)
    assertEquals(0f, blocked.command.linearMps, 0.001f)
    assertEquals(0f, blocked.command.angularRps, 0.001f)
    assertEquals(id, blocked.lockedTrackId)
  }

  @Test
  fun lockedTargetIsKeptAndReacquiredAfterLeaving() {
    val loop = ControlLoop()
    val person = detection(0.30f, 0.10f, 0.60f, 0.90f)
    val other = detection(0.72f, 0.10f, 0.97f, 0.90f)
    val id = loop.step(input(listOf(person), locked = null, nowMs = 0)).tracks.single().id
    var last = loop.step(input(listOf(other), locked = id, nowMs = 10))
    repeat(30) { index ->
      last = loop.step(input(listOf(other), locked = id, nowMs = 20L + index))
    }
    assertEquals(id, last.lockedTrackId)
    assertTrue(last.tracks.any { it.id == id && it.lostFrames > 0 })
    assertEquals(0f, last.command.linearMps, 0.001f)
    assertNotEquals(id, last.tracks.first { it.lostFrames == 0 }.id)

    val back = loop.step(input(listOf(person, other), locked = id, nowMs = 80))
    val recovered = back.tracks.first { it.id == id }
    assertEquals(0, recovered.lostFrames)
    assertEquals(id, back.lockedTrackId)
    assertTrue(back.tracks.any { it.id != id && it.lostFrames == 0 })
  }

  @Test
  fun nearestClassFollowsTheSameLabelAcrossTheFrame() {
    val loop = ControlLoop()
    val start = detection(0.10f, 0.20f, 0.30f, 0.80f)
    val id =
      loop.step(input(listOf(start), locked = null, nowMs = 0, strategy = TrackStrategy.NEAREST_CLASS))
        .tracks
        .single()
        .id
    val far = detection(0.70f, 0.15f, 0.95f, 0.85f)
    val followed =
      loop.step(input(listOf(far), locked = id, nowMs = 50, strategy = TrackStrategy.NEAREST_CLASS))
        .tracks
        .first { it.id == id }
    assertEquals(0, followed.lostFrames)
    assertEquals(far.left, followed.left, 0.001f)
  }

  @Test
  fun motionMatchesAJumpThatIoUMisses() {
    val loop = ControlLoop()
    val start = detection(0.02f, 0.20f, 0.22f, 0.80f)
    val id =
      loop.step(input(listOf(start), locked = null, nowMs = 0, strategy = TrackStrategy.MOTION)).tracks.single().id
    val jumped = detection(0.55f, 0.20f, 0.78f, 0.80f)
    val followed =
      loop.step(input(listOf(jumped), locked = id, nowMs = 50, strategy = TrackStrategy.MOTION))
        .tracks
        .first { it.id == id }
    assertEquals(0, followed.lostFrames)
    assertEquals(jumped.left, followed.left, 0.001f)
  }

  @Test
  fun tapSelectsTheBoxUnderTheFinger() {
    val tracks =
      listOf(
        track(id = 1, left = 0.1f, top = 0.1f, right = 0.4f, bottom = 0.8f),
        track(id = 2, left = 0.6f, top = 0.1f, right = 0.9f, bottom = 0.8f, score = 0.99f),
      )
    assertEquals(1, TargetSelector.trackAt(tracks, 0.2f, 0.4f)!!.id)
    assertNull(TargetSelector.trackAt(tracks, 0.5f, 0.4f))
    val tracker = SortTracker()
    tracker.update(listOf(detection(0.1f, 0.1f, 0.4f, 0.8f)), 1L)
    tracker.reset()
    assertTrue(tracker.update(emptyList(), 2L).isEmpty())
  }

  @Test
  fun metricsUseNearestRankPercentiles() {
    val metrics = FrameMetrics(window = 100)
    for (ms in 1..100) {
      metrics.onProcessed(inferenceDurationNs = ms * 1_000_000L, e2eDurationNs = ms * 1_000_000L, nowNs = ms.toLong())
    }
    metrics.onDrop()
    val snapshot = metrics.snapshot()
    assertEquals(50f, snapshot.inferenceP50Ms, 0.01f)
    assertEquals(95f, snapshot.inferenceP95Ms, 0.01f)
    assertEquals(50f, snapshot.e2eP50Ms, 0.01f)
    assertEquals(1, snapshot.droppedFrames)
    assertEquals(100, snapshot.processedFrames)
    assertEquals(100f / 101f, dropPercent(snapshot.droppedFrames, snapshot.processedFrames), 0.01f)
    assertEquals(0f, dropPercent(0, 0), 0.01f)
    assertEquals(95f, percentileMs((1L..100L).map { it * 1_000_000L }, 95.0), 0.01f)
  }

  @Test
  fun latestSlotKeepsOnlyTheNewestFrame() = runBlocking {
    val dropped = mutableListOf<String>()
    val slot = LatestSlot<String> { dropped.add(it) }
    slot.offer("a")
    slot.offer("b")
    assertEquals(listOf("a"), dropped)
    assertEquals("b", slot.take())
  }

  @Test
  fun tiltAdvicePointsTowardTheLockedObject() {
    val centered = tiltToCenter(0.5f, 0.5f)
    assertTrue(centered.centered)
    assertEquals(0f, centered.yawDeg, 0.01f)
    assertEquals(0f, centered.pitchDeg, 0.01f)

    val right = tiltToCenter(0.75f, 0.5f, horizontalFovDeg = 60f, verticalFovDeg = 80f)
    assertEquals(15f, right.yawDeg, 0.01f)
    assertEquals(0f, right.pitchDeg, 0.01f)
    assertFalse(right.centered)

    val up = tiltToCenter(0.5f, 0.25f, horizontalFovDeg = 60f, verticalFovDeg = 80f)
    assertEquals(-20f, up.pitchDeg, 0.01f)
    assertTrue(up.magnitudeDeg > TiltAdvice.CENTERED_DEADBAND_DEG)
  }

  @Test
  fun personAimSitsOnTheUpperBody() {
    val (aimX, aimY) = aimPoint("person", left = 0.2f, top = 0.1f, right = 0.5f, bottom = 0.9f)
    assertEquals(0.35f, aimX, 0.01f)
    assertEquals(0.244f, aimY, 0.01f)
    val waist = tiltToCenter(0.35f, 0.5f, horizontalFovDeg = 60f, verticalFovDeg = 80f)
    val face = tiltToCenter(aimX, aimY, horizontalFovDeg = 60f, verticalFovDeg = 80f)
    assertTrue(face.pitchDeg < waist.pitchDeg)
  }

  @Test
  fun nonPersonAimStaysAtTheBoxCenter() {
    val (aimX, aimY) = aimPoint("chair", left = 0.2f, top = 0.1f, right = 0.5f, bottom = 0.9f)
    assertEquals(0.35f, aimX, 0.01f)
    assertEquals(0.5f, aimY, 0.01f)
  }

  @Test
  fun peopleModeLocksTheFirstPersonWithoutATap() {
    val loop = ControlLoop()
    val output =
      loop.step(
        input(
          detections = listOf(detection(0.2f, 0.1f, 0.5f, 0.9f), detection(0.6f, 0.1f, 0.9f, 0.9f)),
          locked = null,
          nowMs = 0,
          autoAcquirePerson = true,
        )
      )
    assertEquals(1, output.lockedTrackId)
    assertEquals(1, firstVisiblePerson(output.tracks)?.id)
    val held =
      loop.step(
        input(
          detections =
            listOf(detection(0.22f, 0.1f, 0.52f, 0.9f), detection(0.6f, 0.1f, 0.9f, 0.9f)),
          locked = output.lockedTrackId,
          nowMs = 50,
          autoAcquirePerson = true,
        )
      )
    assertEquals(1, held.lockedTrackId)
    assertEquals(0, held.tracks.first { it.id == 1 }.lostFrames)
  }

  @Test
  fun onlyPersonsDropsOtherLabels() {
    val hits =
      listOf(
        detection(0.2f, 0.1f, 0.4f, 0.8f).copy(label = "person"),
        detection(0.6f, 0.2f, 0.8f, 0.5f).copy(label = "chair"),
      )
    assertEquals(listOf("person"), hits.onlyPersons().map { it.label })
  }

  @Test
  fun overlayMapsLetterboxedTapsIntoTheFrame() {
    val rect = contentRect(viewWidth = 100f, viewHeight = 100f, imageWidth = 50f, imageHeight = 100f)
    assertEquals(25f, rect.left, 0.01f)
    assertEquals(50f, rect.width, 0.01f)
    assertEquals(0f to 0f, viewPointToNormalized(25f, 0f, rect))
    assertNull(viewPointToNormalized(10f, 10f, rect))
  }

  private fun input(
    detections: List<Detection>,
    locked: Int?,
    nowMs: Long,
    telemetry: Telemetry = Telemetry(frontDistanceMm = 2000),
    strategy: TrackStrategy = TrackStrategy.IOU_HOLD,
    autoAcquirePerson: Boolean = false,
  ): ControlInput =
    ControlInput(
      detections = detections,
      timestampNanos = nowMs,
      nowMs = nowMs,
      lockedTrackId = locked,
      telemetry = telemetry,
      seq = nowMs + 1,
      strategy = strategy,
      autoAcquirePerson = autoAcquirePerson,
    )

  private fun command(
    seq: Long,
    timestampMs: Long,
    linear: Float,
    angular: Float = 0f,
  ): VelocityCommand =
    VelocityCommand(
      seq = seq,
      timestampMs = timestampMs,
      linearMps = linear,
      angularRps = angular,
      ttlMs = 200,
    )

  private fun detection(
    left: Float,
    top: Float,
    right: Float,
    bottom: Float,
    score: Float = 0.9f,
  ): Detection = Detection("person", score, left, top, right, bottom)

  private fun track(
    id: Int,
    left: Float,
    top: Float,
    right: Float,
    bottom: Float,
    score: Float = 0.9f,
    lostFrames: Int = 0,
  ) =
    com.google.ai.edge.gallery.rover.tracking.Track(
      id = id,
      label = "person",
      score = score,
      left = left,
      top = top,
      right = right,
      bottom = bottom,
      age = 2,
      hits = 2,
      lostFrames = lostFrames,
      timestampNanos = 0L,
    )
}
