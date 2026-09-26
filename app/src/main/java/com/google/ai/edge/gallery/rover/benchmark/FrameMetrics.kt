/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.benchmark

import kotlin.math.ceil
import kotlinx.coroutines.channels.Channel

data class MetricsSnapshot(
  val fps: Float,
  val droppedFrames: Int,
  val processedFrames: Int,
  val inferenceP50Ms: Float,
  val inferenceP95Ms: Float,
  val e2eP50Ms: Float,
  val e2eP95Ms: Float,
)

class FrameMetrics(private val window: Int = 120) {
  private val lock = Any()
  private val inferenceNs = ArrayDeque<Long>()
  private val e2eNs = ArrayDeque<Long>()
  private var dropped = 0
  private var processed = 0
  private var windowStartNs = 0L
  private var windowFrames = 0
  private var fps = 0f

  fun onDrop() {
    synchronized(lock) { dropped += 1 }
  }

  fun onProcessed(inferenceDurationNs: Long, e2eDurationNs: Long, nowNs: Long) {
    synchronized(lock) {
      processed += 1
      push(inferenceNs, inferenceDurationNs)
      push(e2eNs, e2eDurationNs)
      if (windowStartNs == 0L) windowStartNs = nowNs
      windowFrames += 1
      val elapsed = nowNs - windowStartNs
      if (elapsed >= 1_000_000_000L) {
        fps = windowFrames * 1_000_000_000f / elapsed.toFloat()
        windowFrames = 0
        windowStartNs = nowNs
      }
    }
  }

  fun resetWindow() {
    synchronized(lock) {
      inferenceNs.clear()
      e2eNs.clear()
      windowStartNs = 0L
      windowFrames = 0
      fps = 0f
    }
  }

  fun snapshot(): MetricsSnapshot {
    return synchronized(lock) {
      MetricsSnapshot(
        fps = fps,
        droppedFrames = dropped,
        processedFrames = processed,
        inferenceP50Ms = percentileMs(inferenceNs, 50.0),
        inferenceP95Ms = percentileMs(inferenceNs, 95.0),
        e2eP50Ms = percentileMs(e2eNs, 50.0),
        e2eP95Ms = percentileMs(e2eNs, 95.0),
      )
    }
  }

  private fun push(samples: ArrayDeque<Long>, value: Long) {
    samples.addLast(value)
    while (samples.size > window) samples.removeFirst()
  }
}

/** Share of camera frames the detector skipped, as 0..100. */
fun dropPercent(dropped: Int, processed: Int): Float {
  val total = dropped + processed
  if (total <= 0) return 0f
  return dropped * 100f / total.toFloat()
}

fun percentileMs(samples: Collection<Long>, percentile: Double): Float {
  if (samples.isEmpty()) return 0f
  val sorted = samples.sorted()
  val rank = ceil(percentile / 100.0 * sorted.size).toInt().coerceIn(1, sorted.size)
  return sorted[rank - 1] / 1_000_000f
}

/**
 * Holds the newest item and drops the previous one. The camera analyzer uses this
 * so a slow detector never builds a frame queue.
 */
class LatestSlot<T>(private val onDrop: (T) -> Unit) {
  private val lock = Any()
  private var value: T? = null
  private val signal = Channel<Unit>(Channel.CONFLATED)

  fun offer(item: T) {
    val dropped =
      synchronized(lock) {
        val previous = value
        value = item
        previous
      }
    if (dropped != null) onDrop(dropped)
    signal.trySend(Unit)
  }

  suspend fun take(): T {
    while (true) {
      val item =
        synchronized(lock) {
          val current = value
          value = null
          current
        }
      if (item != null) return item
      signal.receive()
    }
  }
}
