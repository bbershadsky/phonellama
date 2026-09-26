/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.tracking

import com.google.ai.edge.gallery.rover.inference.Detection
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class Track(
  val id: Int,
  val label: String,
  val score: Float,
  val left: Float,
  val top: Float,
  val right: Float,
  val bottom: Float,
  val age: Int,
  val hits: Int,
  val lostFrames: Int,
  val timestampNanos: Long,
) {
  val centerX: Float
    get() = (left + right) / 2f

  val centerY: Float
    get() = (top + bottom) / 2f

  val width: Float
    get() = (right - left).coerceAtLeast(0f)

  val height: Float
    get() = (bottom - top).coerceAtLeast(0f)
}

/** Lowest id among people currently in frame. That is the first person the tracker created. */
fun firstVisiblePerson(tracks: List<Track>): Track? =
  tracks.filter { it.lostFrames == 0 && it.label.equals("person", ignoreCase = true) }.minByOrNull { it.id }

enum class TrackStrategy(val label: String) {
  /** Keep the id and only join a box that still overlaps or sits near the last one. */
  IOU_HOLD("IoU hold"),

  /** While locked, follow the closest detection with the same label anywhere in the frame. */
  NEAREST_CLASS("Nearest class"),

  /** Match by predicted center distance instead of box overlap, so a jumping box can stay locked. */
  MOTION("Motion predict"),
}

interface ObjectTracker {
  fun update(detections: List<Detection>, timestampNanos: Long): List<Track>

  fun pin(trackId: Int?) {}

  fun setStrategy(strategy: TrackStrategy) {}

  fun reset()
}

/**
 * Greedy IoU tracker with a one-step constant-velocity prediction.
 * Identity stays on the overlapping box, so a second person nearby does not inherit the id
 * while the original box can still be matched.
 */
class SortTracker(
  private val iouThreshold: Float = 0.3f,
  private val maxLostFrames: Int = 20,
) : ObjectTracker {
  private var nextId = 1
  private var pinnedId: Int? = null
  private var strategy = TrackStrategy.IOU_HOLD
  private val tracks = mutableListOf<MutableTrack>()

  override fun pin(trackId: Int?) {
    pinnedId = trackId
  }

  override fun setStrategy(strategy: TrackStrategy) {
    this.strategy = strategy
  }

  override fun update(detections: List<Detection>, timestampNanos: Long): List<Track> {
    tracks.forEach { it.predict() }
    val usedTracks = HashSet<Int>()
    val usedDetections = HashSet<Int>()
    when (strategy) {
      TrackStrategy.NEAREST_CLASS -> {
        claimNearestSameLabel(detections, usedTracks, usedDetections, timestampNanos)
        matchByIou(detections, usedTracks, usedDetections, timestampNanos)
      }
      TrackStrategy.MOTION -> matchByDistance(detections, usedTracks, usedDetections, timestampNanos)
      TrackStrategy.IOU_HOLD -> {
        matchByIou(detections, usedTracks, usedDetections, timestampNanos)
        reacquirePinned(detections, usedTracks, usedDetections, timestampNanos)
      }
    }
    for (trackIndex in tracks.indices) {
      if (trackIndex !in usedTracks) tracks[trackIndex].markMissed(timestampNanos)
    }
    tracks.removeAll { it.lostFrames > maxLostFrames && it.id != pinnedId }
    for (detectionIndex in detections.indices) {
      if (detectionIndex in usedDetections) continue
      tracks.add(MutableTrack(nextId++, detections[detectionIndex], timestampNanos))
    }
    return tracks.map { it.toTrack() }
  }

  override fun reset() {
    tracks.clear()
    nextId = 1
    pinnedId = null
  }

  private fun matchByIou(
    detections: List<Detection>,
    usedTracks: MutableSet<Int>,
    usedDetections: MutableSet<Int>,
    timestampNanos: Long,
  ) {
    val pairs = mutableListOf<Triple<Int, Int, Float>>()
    for (trackIndex in tracks.indices) {
      if (trackIndex in usedTracks) continue
      val track = tracks[trackIndex]
      for (detectionIndex in detections.indices) {
        if (detectionIndex in usedDetections) continue
        val overlap =
          boxIou(track.left, track.top, track.right, track.bottom, detections[detectionIndex])
        if (overlap >= iouThreshold) pairs.add(Triple(trackIndex, detectionIndex, overlap))
      }
    }
    pairs.sortByDescending { it.third }
    for ((trackIndex, detectionIndex, _) in pairs) {
      if (trackIndex in usedTracks || detectionIndex in usedDetections) continue
      usedTracks.add(trackIndex)
      usedDetections.add(detectionIndex)
      tracks[trackIndex].update(detections[detectionIndex], timestampNanos)
    }
  }

  private fun matchByDistance(
    detections: List<Detection>,
    usedTracks: MutableSet<Int>,
    usedDetections: MutableSet<Int>,
    timestampNanos: Long,
  ) {
    val pairs = mutableListOf<Triple<Int, Int, Float>>()
    for (trackIndex in tracks.indices) {
      val track = tracks[trackIndex]
      val centerX = (track.left + track.right) / 2f
      val centerY = (track.top + track.bottom) / 2f
      val pinned = track.id == pinnedId
      for (detectionIndex in detections.indices) {
        val detection = detections[detectionIndex]
        val distance = hypot(centerX - detection.centerX, centerY - detection.centerY)
        val sameLabel = track.label == detection.label
        val limit = if (pinned && sameLabel) 0.75f else 0.4f
        if (distance > limit) continue
        val rank = if (pinned && sameLabel) distance - 0.25f else distance
        pairs.add(Triple(trackIndex, detectionIndex, rank))
      }
    }
    pairs.sortBy { it.third }
    for ((trackIndex, detectionIndex, _) in pairs) {
      if (trackIndex in usedTracks || detectionIndex in usedDetections) continue
      usedTracks.add(trackIndex)
      usedDetections.add(detectionIndex)
      tracks[trackIndex].update(detections[detectionIndex], timestampNanos)
    }
  }

  /** Snap the locked track onto the closest same-label box, even if it jumped across the frame. */
  private fun claimNearestSameLabel(
    detections: List<Detection>,
    usedTracks: MutableSet<Int>,
    usedDetections: MutableSet<Int>,
    timestampNanos: Long,
  ) {
    val pinnedIndex = tracks.indexOfFirst { it.id == pinnedId }
    if (pinnedIndex < 0) return
    val pinned = tracks[pinnedIndex]
    val centerX = (pinned.left + pinned.right) / 2f
    val centerY = (pinned.top + pinned.bottom) / 2f
    var bestIndex = -1
    var bestDistance = Float.MAX_VALUE
    for (detectionIndex in detections.indices) {
      val detection = detections[detectionIndex]
      if (detection.label != pinned.label) continue
      val distance = hypot(centerX - detection.centerX, centerY - detection.centerY)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = detectionIndex
      }
    }
    if (bestIndex < 0) return
    usedTracks.add(pinnedIndex)
    usedDetections.add(bestIndex)
    pinned.update(detections[bestIndex], timestampNanos)
  }

  /**
   * After a miss, attach the pinned track to the same label near its last position.
   * A detection already claimed by another live track is left alone.
   */
  private fun reacquirePinned(
    detections: List<Detection>,
    usedTracks: MutableSet<Int>,
    usedDetections: MutableSet<Int>,
    timestampNanos: Long,
  ) {
    val pinnedIndex = tracks.indexOfFirst { it.id == pinnedId }
    if (pinnedIndex < 0 || pinnedIndex in usedTracks) return
    val pinned = tracks[pinnedIndex]
    var bestIndex = -1
    var bestDistance = Float.MAX_VALUE
    for (detectionIndex in detections.indices) {
      if (detectionIndex in usedDetections) continue
      val distance = reacquireDistance(pinned, detections[detectionIndex]) ?: continue
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = detectionIndex
      }
    }
    if (bestIndex < 0) return
    usedTracks.add(pinnedIndex)
    usedDetections.add(bestIndex)
    pinned.update(detections[bestIndex], timestampNanos)
  }

  private fun reacquireDistance(track: MutableTrack, detection: Detection): Float? {
    if (track.label != detection.label) return null
    val centerX = (track.left + track.right) / 2f
    val centerY = (track.top + track.bottom) / 2f
    val distance = hypot(centerX - detection.centerX, centerY - detection.centerY)
    val limit = if (track.lostFrames == 0) 0.30f else 0.48f
    if (distance > limit) return null
    return distance
  }

  private class MutableTrack(
    val id: Int,
    detection: Detection,
    timestampNanos: Long,
  ) {
    var label: String = detection.label
    var score: Float = detection.score
    var left: Float = detection.left
    var top: Float = detection.top
    var right: Float = detection.right
    var bottom: Float = detection.bottom
    var vx: Float = 0f
    var vy: Float = 0f
    var age: Int = 1
    var hits: Int = 1
    var lostFrames: Int = 0
    var timestampNanos: Long = timestampNanos

    fun predict() {
      left += vx
      right += vx
      top += vy
      bottom += vy
      val centerX = (left + right) / 2f
      val centerY = (top + bottom) / 2f
      val clampX = centerX.coerceIn(-0.15f, 1.15f) - centerX
      val clampY = centerY.coerceIn(-0.15f, 1.15f) - centerY
      left += clampX
      right += clampX
      top += clampY
      bottom += clampY
      if (lostFrames > 0) {
        vx *= 0.85f
        vy *= 0.85f
      }
    }

    fun update(detection: Detection, timestampNanos: Long) {
      val predictedCenterX = (left + right) / 2f
      val predictedCenterY = (top + bottom) / 2f
      vx = vx * 0.3f + (detection.centerX - predictedCenterX) * 0.7f
      vy = vy * 0.3f + (detection.centerY - predictedCenterY) * 0.7f
      label = detection.label
      score = detection.score
      left = detection.left
      top = detection.top
      right = detection.right
      bottom = detection.bottom
      lostFrames = 0
      hits += 1
      age += 1
      this.timestampNanos = timestampNanos
      nudgeOffEdge()
    }

    fun markMissed(timestampNanos: Long) {
      lostFrames += 1
      age += 1
      this.timestampNanos = timestampNanos
      nudgeOffEdge()
    }

    private fun nudgeOffEdge() {
      val edge = 0.12f
      val spansWidth = left <= edge && right >= 1f - edge
      val spansHeight = top <= edge && bottom >= 1f - edge
      if (!spansWidth) {
        if (right >= 1f - edge && left > edge) vx = max(vx, 0.04f)
        if (left <= edge && right < 1f - edge) vx = min(vx, -0.04f)
      }
      if (!spansHeight) {
        if (bottom >= 1f - edge && top > edge) vy = max(vy, 0.04f)
        if (top <= edge && bottom < 1f - edge) vy = min(vy, -0.04f)
      }
    }

    fun toTrack(): Track =
      Track(
        id = id,
        label = label,
        score = score,
        left = left,
        top = top,
        right = right,
        bottom = bottom,
        age = age,
        hits = hits,
        lostFrames = lostFrames,
        timestampNanos = timestampNanos,
      )
  }
}

object TargetSelector {
  /** Track whose current box contains the tap, preferring the higher score. */
  fun trackAt(tracks: List<Track>, normalizedX: Float, normalizedY: Float): Track? {
    return tracks
      .asSequence()
      .filter { it.lostFrames == 0 }
      .filter { normalizedX in it.left..it.right && normalizedY in it.top..it.bottom }
      .maxByOrNull { it.score }
  }
}

private fun boxIou(
  left: Float,
  top: Float,
  right: Float,
  bottom: Float,
  detection: Detection,
): Float {
  val overlapLeft = max(left, detection.left)
  val overlapTop = max(top, detection.top)
  val overlapRight = min(right, detection.right)
  val overlapBottom = min(bottom, detection.bottom)
  val overlapWidth = (overlapRight - overlapLeft).coerceAtLeast(0f)
  val overlapHeight = (overlapBottom - overlapTop).coerceAtLeast(0f)
  val intersection = overlapWidth * overlapHeight
  val trackArea = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
  val union = trackArea + detection.area - intersection
  if (union <= 0f) return 0f
  return intersection / union
}
