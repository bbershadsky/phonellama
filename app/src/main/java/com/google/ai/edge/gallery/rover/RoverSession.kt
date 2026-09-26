/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.rover.benchmark.FrameMetrics
import com.google.ai.edge.gallery.rover.benchmark.LatestSlot
import com.google.ai.edge.gallery.rover.control.ControlInput
import com.google.ai.edge.gallery.rover.control.ControlLoop
import com.google.ai.edge.gallery.rover.inference.Detection
import com.google.ai.edge.gallery.rover.inference.DetectorModel
import com.google.ai.edge.gallery.rover.inference.ImageFrame
import com.google.ai.edge.gallery.rover.inference.MediaPipeObjectDetectorBackend
import com.google.ai.edge.gallery.rover.inference.onlyPersons
import com.google.ai.edge.gallery.rover.inference.UnavailableVisionBackend
import com.google.ai.edge.gallery.rover.inference.VisionBackend
import com.google.ai.edge.gallery.rover.protocol.Telemetry
import com.google.ai.edge.gallery.rover.protocol.VelocityCommand
import com.google.ai.edge.gallery.rover.safety.StopReason
import com.google.ai.edge.gallery.rover.tracking.Track
import com.google.ai.edge.gallery.rover.tracking.TrackStrategy
import com.google.ai.edge.gallery.rover.transport.MockRobotTransport
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger

data class RoverUiState(
  val status: String,
  val tracks: List<Track> = emptyList(),
  val lockedTrackId: Int? = null,
  val reason: StopReason = StopReason.NO_TARGET,
  val command: VelocityCommand? = null,
  val leftWheelMps: Float = 0f,
  val rightWheelMps: Float = 0f,
  val fps: Float = 0f,
  val droppedFrames: Int = 0,
  val processedFrames: Int = 0,
  val inferenceP50Ms: Float = 0f,
  val inferenceP95Ms: Float = 0f,
  val e2eP50Ms: Float = 0f,
  val e2eP95Ms: Float = 0f,
  val frameWidth: Int = 0,
  val frameHeight: Int = 0,
  val frontDistanceMm: Int = 2000,
  val estop: Boolean = false,
  val strategy: TrackStrategy = TrackStrategy.NEAREST_CLASS,
  val personsOnly: Boolean = true,
  val detector: DetectorModel = DetectorModel.LITE0,
)

/**
 * Detector, tracker, follow controller, and mock transport.
 * The camera thread only publishes the latest frame. This worker does the rest.
 */
class RoverSession(
  initialBackend: VisionBackend,
  private val appContext: Context,
) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
  private val detectMutex = Mutex()
  @Volatile private var backend: VisionBackend = initialBackend
  private val transport = MockRobotTransport()
  private val control = ControlLoop()
  private val metrics = FrameMetrics()
  private val lockedTrackId = AtomicReference<Int?>(null)
  private val strategy = AtomicReference(TrackStrategy.NEAREST_CLASS)
  private val personsOnly = AtomicBoolean(true)
  private val detectorModel = AtomicReference(DetectorModel.LITE0)
  private val detectorGeneration = AtomicInteger(0)
  private val loadingDetector = AtomicBoolean(false)
  private val loadError = AtomicReference<String?>(null)
  private val telemetry = AtomicReference(Telemetry(frontDistanceMm = 2000))
  private val sequence = AtomicLong(0)
  private val closed = AtomicBoolean(false)
  private val facingFront = AtomicBoolean(false)
  private val resetTracking = AtomicBoolean(false)
  private val frames =
    LatestSlot<ImageFrame> { dropped ->
      metrics.onDrop()
      dropped.release()
    }
  private val _state = MutableStateFlow(RoverUiState(status = statusText()))
  val state: StateFlow<RoverUiState> = _state.asStateFlow()
  private val job: Job

  init {
    job =
      scope.launch {
        transport.connect()
        try {
          while (true) {
            val frame = frames.take()
            try {
              process(frame)
            } finally {
              frame.release()
            }
          }
        } finally {
          transport.disconnect()
        }
      }
  }

  fun offer(bitmap: Bitmap, receivedNs: Long) {
    if (closed.get()) {
      if (!bitmap.isRecycled) bitmap.recycle()
      return
    }
    frames.offer(ImageFrame(bitmap, receivedNs))
  }

  fun selectTrack(id: Int) {
    lockedTrackId.set(id)
  }

  fun clearTarget() {
    lockedTrackId.set(null)
  }

  fun setStrategy(value: TrackStrategy) {
    strategy.set(value)
    _state.value = _state.value.copy(strategy = value)
  }

  /**
   * Loads [model] on the worker and swaps it in after the current frame.
   * The previous detector keeps running until the new one is ready.
   */
  fun setDetector(model: DetectorModel) {
    if (closed.get() || model == detectorModel.get()) return
    val previous = detectorModel.get()
    val generation = detectorGeneration.incrementAndGet()
    detectorModel.set(model)
    loadingDetector.set(true)
    loadError.set(null)
    resetTracking.set(true)
    lockedTrackId.set(null)
    _state.value =
      _state.value.copy(
        detector = model,
        status = statusText(),
        tracks = emptyList(),
        lockedTrackId = null,
        reason = StopReason.NO_TARGET,
      )
    scope.launch {
      val next = MediaPipeObjectDetectorBackend.create(appContext, model)
      withContext(NonCancellable) {
        if (generation != detectorGeneration.get() || closed.get()) {
          next.close()
          return@withContext
        }
        if (next is UnavailableVisionBackend) {
          next.close()
          detectorModel.set(previous)
          loadingDetector.set(false)
          loadError.set("${model.shortLabel} failed to load")
          _state.value = _state.value.copy(detector = previous, status = statusText())
          return@withContext
        }
        val installed =
          detectMutex.withLock {
            if (generation != detectorGeneration.get() || closed.get()) {
              false
            } else {
              val old = backend
              backend = next
              old.close()
              loadingDetector.set(false)
              true
            }
          }
        if (!installed) {
          next.close()
          return@withContext
        }
        metrics.resetWindow()
        _state.value = _state.value.copy(detector = model, status = statusText())
      }
    }
  }

  /** When enabled, chairs and other COCO classes never enter the tracker. */

  fun setPersonsOnly(enabled: Boolean) {
    personsOnly.set(enabled)
    val tracks = visibleTracks(_state.value.tracks, enabled)
    val locked = lockedTrackId.get()
    if (locked != null && tracks.none { it.id == locked }) lockedTrackId.set(null)
    _state.value =
      _state.value.copy(personsOnly = enabled, tracks = tracks, lockedTrackId = lockedTrackId.get())
  }

  /** Drops the current tracks when the lens changes so ids are not reused across cameras. */
  fun setFacingFront(front: Boolean) {
    facingFront.set(front)
    resetTracking.set(true)
    lockedTrackId.set(null)
    _state.value =
      _state.value.copy(
        status = statusText(),
        tracks = emptyList(),
        lockedTrackId = null,
        reason = StopReason.NO_TARGET,
      )
  }

  fun setEstop(enabled: Boolean) {
    telemetry.updateAndGet { it.copy(estop = enabled) }
    _state.value = _state.value.copy(estop = enabled)
  }

  fun setFrontDistanceMm(distanceMm: Int) {
    val clamped = distanceMm.coerceIn(0, 5000)
    telemetry.updateAndGet { it.copy(frontDistanceMm = clamped) }
    _state.value = _state.value.copy(frontDistanceMm = clamped)
  }

  fun close() {
    if (!closed.compareAndSet(false, true)) return
    scope.cancel()
    runBlocking {
      job.join()
      detectMutex.withLock { backend.close() }
    }
  }

  private suspend fun process(frame: ImageFrame) {
    if (resetTracking.compareAndSet(true, false)) control.reset()
    val inferenceStartNs = System.nanoTime()
    val detections =
      try {
        val hits = detectMutex.withLock { backend.detect(frame) }
        if (personsOnly.get()) hits.onlyPersons() else hits
      } catch (error: Exception) {
        Log.e(TAG, "detect failed", error)
        emptyList()
      }
    val inferenceEndNs = System.nanoTime()
    val nowMs = System.currentTimeMillis()
    val requestedLock = lockedTrackId.get()
    val output =
      control.step(
        ControlInput(
          detections = detections,
          timestampNanos = frame.receivedNs,
          nowMs = nowMs,
          lockedTrackId = requestedLock,
          telemetry = telemetry.get(),
          seq = sequence.incrementAndGet(),
          strategy = strategy.get(),
          autoAcquirePerson = personsOnly.get(),
        )
      )
    if (personsOnly.get()) {
      val acquired = output.lockedTrackId
      if (acquired != null) lockedTrackId.compareAndSet(null, acquired)
    }
    transport.send(output.command)
    val sentNs = System.nanoTime()
    metrics.onProcessed(
      inferenceDurationNs = inferenceEndNs - inferenceStartNs,
      e2eDurationNs = sentNs - frame.receivedNs,
      nowNs = sentNs,
    )
    val peopleOnly = personsOnly.get()
    val shown = visibleTracks(output.tracks, peopleOnly)
    val locked = lockedTrackId.get()
    if (locked != null && shown.none { it.id == locked }) lockedTrackId.set(null)
    val snapshot = metrics.snapshot()
    val currentTelemetry = telemetry.get()
    _state.value =
      RoverUiState(
        status = statusText(),
        tracks = shown,
        lockedTrackId = lockedTrackId.get(),
        reason = output.reason,
        command = output.command,
        leftWheelMps = output.leftWheelMps,
        rightWheelMps = output.rightWheelMps,
        fps = snapshot.fps,
        droppedFrames = snapshot.droppedFrames,
        processedFrames = snapshot.processedFrames,
        inferenceP50Ms = snapshot.inferenceP50Ms,
        inferenceP95Ms = snapshot.inferenceP95Ms,
        e2eP50Ms = snapshot.e2eP50Ms,
        e2eP95Ms = snapshot.e2eP95Ms,
        frameWidth = frame.bitmap.width,
        frameHeight = frame.bitmap.height,
        frontDistanceMm = currentTelemetry.frontDistanceMm ?: 0,
        estop = currentTelemetry.estop,
        strategy = strategy.get(),
        personsOnly = peopleOnly,
        detector = detectorModel.get(),
      )
  }

  private fun statusText(): String {
    val lens = if (facingFront.get()) "Front camera" else "Rear camera"
    val model = detectorModel.get()
    val error = loadError.get()
    val unavailable = backend as? UnavailableVisionBackend
    return when {
      loadingDetector.get() -> "$lens · loading ${model.shortLabel}"
      error != null -> "$lens · $error · ${model.shortLabel}"
      unavailable != null -> "$lens · detector unavailable: ${unavailable.reason}"
      else -> "$lens · ${model.label} · mock motors"
    }
  }

  private companion object {
    const val TAG = "RoverVision"
  }
}

private fun visibleTracks(tracks: List<Track>, personsOnly: Boolean): List<Track> =
  if (personsOnly) tracks.filter { it.label.equals(Detection.PERSON_LABEL, ignoreCase = true) } else tracks

private fun ImageFrame.release() {
  if (!bitmap.isRecycled) bitmap.recycle()
}
