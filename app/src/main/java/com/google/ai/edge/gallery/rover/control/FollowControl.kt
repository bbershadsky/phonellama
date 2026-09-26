/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.control

import com.google.ai.edge.gallery.rover.protocol.Telemetry
import com.google.ai.edge.gallery.rover.protocol.VelocityCommand
import com.google.ai.edge.gallery.rover.protocol.wheelSpeeds
import com.google.ai.edge.gallery.rover.safety.RoverConfig
import com.google.ai.edge.gallery.rover.safety.SafetyArbiter
import com.google.ai.edge.gallery.rover.safety.StopReason
import com.google.ai.edge.gallery.rover.tracking.ObjectTracker
import com.google.ai.edge.gallery.rover.tracking.SortTracker
import com.google.ai.edge.gallery.rover.tracking.Track
import com.google.ai.edge.gallery.rover.tracking.TrackStrategy
import com.google.ai.edge.gallery.rover.tracking.firstVisiblePerson
import com.google.ai.edge.gallery.rover.inference.Detection
import kotlin.math.abs

data class RawVelocity(val linearMps: Float, val angularRps: Float)

/**
 * Image-plane follow law.
 * Positive angular velocity turns left (counterclockwise), matching [wheelSpeeds].
 * A target right of center therefore produces a negative angular velocity.
 */
class FollowController(private val config: RoverConfig = RoverConfig()) {
  fun compute(track: Track): RawVelocity {
    var horizontalError = track.centerX - 0.5f
    if (abs(horizontalError) < config.deadbandX) horizontalError = 0f
    var distanceError = config.desiredWidth - track.width
    if (abs(distanceError) < config.deadbandWidth) distanceError = 0f
    var linear = config.kd * distanceError
    var angular = -config.kx * horizontalError
    if (abs(horizontalError) > config.edgeError) linear *= config.edgeLinearScale
    linear = linear.coerceIn(-config.maxLinearMps * config.reverseScale, config.maxLinearMps)
    angular = angular.coerceIn(-config.maxAngularRps, config.maxAngularRps)
    return RawVelocity(linear, angular)
  }
}

/** Limits how fast the command can change. A reset returns to a full stop. */
class MotionSmoother(private val config: RoverConfig = RoverConfig()) {
  private var linear = 0f
  private var angular = 0f
  private var lastMs = -1L

  fun smooth(target: RawVelocity, nowMs: Long): RawVelocity {
    if (lastMs < 0L) {
      lastMs = nowMs
      linear = 0f
      angular = 0f
      return RawVelocity(0f, 0f)
    }
    val dtSec = ((nowMs - lastMs).coerceIn(0L, 100L)) / 1000f
    lastMs = nowMs
    linear = approach(linear, target.linearMps, config.maxLinearAccelMps2 * dtSec)
    angular = approach(angular, target.angularRps, config.maxAngularAccelRps2 * dtSec)
    return RawVelocity(linear, angular)
  }

  fun reset() {
    linear = 0f
    angular = 0f
    lastMs = -1L
  }

  private fun approach(current: Float, target: Float, maxDelta: Float): Float {
    val delta = (target - current).coerceIn(-maxDelta, maxDelta)
    return current + delta
  }
}

data class ControlInput(
  val detections: List<Detection>,
  val timestampNanos: Long,
  val nowMs: Long,
  val lockedTrackId: Int?,
  val telemetry: Telemetry,
  val seq: Long,
  val strategy: TrackStrategy = TrackStrategy.IOU_HOLD,
  val autoAcquirePerson: Boolean = false,
)

data class ControlOutput(
  val tracks: List<Track>,
  val lockedTrackId: Int?,
  val reason: StopReason,
  val command: VelocityCommand,
  val leftWheelMps: Float,
  val rightWheelMps: Float,
)

class ControlLoop(
  private val config: RoverConfig = RoverConfig(),
  private val tracker: ObjectTracker = SortTracker(),
  private val controller: FollowController = FollowController(config),
  private val smoother: MotionSmoother = MotionSmoother(config),
  private val safety: SafetyArbiter = SafetyArbiter(),
) {
  fun step(input: ControlInput): ControlOutput {
    tracker.setStrategy(input.strategy)
    var lockId = input.lockedTrackId
    tracker.pin(lockId)
    val tracks = tracker.update(input.detections, input.timestampNanos)
    if (input.autoAcquirePerson && lockId == null) {
      lockId = firstVisiblePerson(tracks)?.id
      if (lockId != null) tracker.pin(lockId)
    }
    val locked = tracks.find { it.id == lockId }
    val lockHeld = lockId != null
    val reason = safety.evaluate(locked, lockHeld, input.telemetry, config)
    val velocity =
      if (reason == StopReason.NONE && locked != null) {
        smoother.smooth(controller.compute(locked), input.nowMs)
      } else {
        smoother.reset()
        RawVelocity(0f, 0f)
      }
    val (left, right) = wheelSpeeds(velocity.linearMps, velocity.angularRps, config.wheelSeparationM)
    val command =
      VelocityCommand(
        seq = input.seq,
        timestampMs = input.nowMs,
        linearMps = velocity.linearMps,
        angularRps = velocity.angularRps,
        ttlMs = config.commandTtlMs,
      )
    return ControlOutput(
      tracks = tracks,
      lockedTrackId = if (lockHeld) lockId else null,
      reason = reason,
      command = command,
      leftWheelMps = left,
      rightWheelMps = right,
    )
  }

  fun reset() {
    tracker.reset()
    smoother.reset()
  }
}
