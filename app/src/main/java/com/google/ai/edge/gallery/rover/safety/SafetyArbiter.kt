/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.safety

import com.google.ai.edge.gallery.rover.protocol.Telemetry
import com.google.ai.edge.gallery.rover.tracking.Track

enum class StopReason {
  NONE,
  NO_TARGET,
  LOST_TARGET,
  LOW_CONFIDENCE,
  OBSTACLE,
  SENSOR_FAULT,
  LOW_BATTERY,
  ESTOP,
}

data class RoverConfig(
  val kx: Float = 1.6f,
  val kd: Float = 1.2f,
  val desiredWidth: Float = 0.35f,
  val deadbandX: Float = 0.04f,
  val deadbandWidth: Float = 0.03f,
  val maxLinearMps: Float = 0.35f,
  val reverseScale: Float = 0.4f,
  val maxAngularRps: Float = 0.8f,
  val edgeError: Float = 0.25f,
  val edgeLinearScale: Float = 0.3f,
  val minConfidence: Float = 0.6f,
  val obstacleStopMm: Int = 400,
  val minBatteryV: Float = 10.5f,
  val wheelSeparationM: Float = 0.22f,
  val commandTtlMs: Int = 200,
  val maxLinearAccelMps2: Float = 0.6f,
  val maxAngularAccelRps2: Float = 2.0f,
)

/**
 * Decides whether a follow command is allowed. Higher-priority faults win, and a fault
 * zeros velocity even when the detector still has a target.
 */
class SafetyArbiter {
  fun evaluate(
    track: Track?,
    hasLock: Boolean,
    telemetry: Telemetry,
    config: RoverConfig,
  ): StopReason {
    if (telemetry.estop) return StopReason.ESTOP
    if (telemetry.fault) return StopReason.SENSOR_FAULT
    if (telemetry.batteryV < config.minBatteryV) return StopReason.LOW_BATTERY
    val distanceMm = telemetry.frontDistanceMm
    if (distanceMm != null && distanceMm <= config.obstacleStopMm) return StopReason.OBSTACLE
    if (!hasLock) return StopReason.NO_TARGET
    if (track == null || track.lostFrames > 0) return StopReason.LOST_TARGET
    if (track.score < config.minConfidence) return StopReason.LOW_CONFIDENCE
    return StopReason.NONE
  }
}
