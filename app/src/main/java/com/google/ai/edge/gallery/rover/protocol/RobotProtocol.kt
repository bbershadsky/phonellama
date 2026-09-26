/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class VelocityCommand(
  val type: String = "cmd_vel",
  val seq: Long,
  @SerialName("timestamp_ms") val timestampMs: Long,
  @SerialName("linear_mps") val linearMps: Float,
  @SerialName("angular_rps") val angularRps: Float,
  @SerialName("ttl_ms") val ttlMs: Int,
)

@Serializable
data class Telemetry(
  val type: String = "telemetry",
  val seq: Long = 0,
  @SerialName("left_ticks") val leftTicks: Int = 0,
  @SerialName("right_ticks") val rightTicks: Int = 0,
  @SerialName("battery_v") val batteryV: Float = 12.6f,
  @SerialName("front_distance_mm") val frontDistanceMm: Int? = null,
  val fault: Boolean = false,
  val estop: Boolean = false,
)

object JsonProtocol {
  private val json = Json {
    encodeDefaults = true
    ignoreUnknownKeys = true
    explicitNulls = false
  }

  fun encode(command: VelocityCommand): String =
    json.encodeToString(VelocityCommand.serializer(), command)

  fun decodeCommand(text: String): VelocityCommand =
    json.decodeFromString(VelocityCommand.serializer(), text)

  fun encode(telemetry: Telemetry): String = json.encodeToString(Telemetry.serializer(), telemetry)

  fun decodeTelemetry(text: String): Telemetry = json.decodeFromString(Telemetry.serializer(), text)
}

/**
 * Accepts a command only when its sequence is newer than the last accepted command
 * and its age is inside the TTL. This is the phone-side stand-in for the ESP32 watchdog.
 */
class CommandGate {
  private var lastSeq: Long = -1

  fun accept(command: VelocityCommand, nowMs: Long): Boolean {
    if (command.seq <= lastSeq) return false
    val ageMs = nowMs - command.timestampMs
    if (ageMs > command.ttlMs || ageMs < -command.ttlMs) return false
    lastSeq = command.seq
    return true
  }

  fun reset() {
    lastSeq = -1
  }
}

/** Differential drive. Positive angular velocity is counterclockwise (left turn). */
fun wheelSpeeds(linearMps: Float, angularRps: Float, wheelSeparationM: Float): Pair<Float, Float> {
  val half = angularRps * wheelSeparationM / 2f
  return (linearMps - half) to (linearMps + half)
}
