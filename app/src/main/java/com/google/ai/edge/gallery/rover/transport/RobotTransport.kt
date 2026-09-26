/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.transport

import com.google.ai.edge.gallery.rover.protocol.CommandGate
import com.google.ai.edge.gallery.rover.protocol.Telemetry
import com.google.ai.edge.gallery.rover.protocol.VelocityCommand
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

interface RobotTransport {
  suspend fun connect()

  suspend fun send(command: VelocityCommand)

  fun telemetry(): Flow<Telemetry>

  suspend fun disconnect()
}

/**
 * Records velocity commands and drops them when the TTL expires.
 * No motor driver is attached.
 */
class MockRobotTransport(
  private val gate: CommandGate = CommandGate(),
) : RobotTransport {
  private val telemetryState = MutableStateFlow(Telemetry())
  private var connected = false
  private var lastAccepted: VelocityCommand? = null
  var rejectedCount: Int = 0
    private set

  override suspend fun connect() {
    connected = true
  }

  override suspend fun send(command: VelocityCommand) {
    offer(command, command.timestampMs)
  }

  fun offer(command: VelocityCommand, nowMs: Long): Boolean {
    if (!connected) {
      rejectedCount += 1
      return false
    }
    if (!gate.accept(command, nowMs)) {
      rejectedCount += 1
      return false
    }
    lastAccepted = command
    return true
  }

  /** Command the motors would still be honoring. Zero after the TTL. */
  fun effective(nowMs: Long): VelocityCommand? {
    val command = lastAccepted ?: return null
    val ageMs = nowMs - command.timestampMs
    if (ageMs > command.ttlMs) {
      return command.copy(linearMps = 0f, angularRps = 0f)
    }
    return command
  }

  fun updateTelemetry(telemetry: Telemetry) {
    telemetryState.value = telemetry
  }

  override fun telemetry(): Flow<Telemetry> = telemetryState.asStateFlow()

  override suspend fun disconnect() {
    connected = false
    lastAccepted = null
  }
}
