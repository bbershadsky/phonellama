/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.ui

import kotlin.math.hypot

/**
 * How far to tilt the phone so the locked box center meets the frame center.
 * Positive yaw tilts right. Positive pitch tilts down.
 * Field of view is an approximate portrait phone camera, not a calibrated value.
 */
data class TiltAdvice(
  val yawDeg: Float,
  val pitchDeg: Float,
) {
  val magnitudeDeg: Float
    get() = hypot(yawDeg, pitchDeg)

  val centered: Boolean
    get() = magnitudeDeg < CENTERED_DEADBAND_DEG

  companion object {
    const val CENTERED_DEADBAND_DEG = 3f
    const val HORIZONTAL_FOV_DEG = 60f
    const val VERTICAL_FOV_DEG = 80f
  }
}

private const val PERSON_UPPER_BODY_FRACTION = 0.18f

/**
 * Point the tilt arrow should chase.
 * A person box from EfficientDet covers the whole body, so the geometric center sits near the waist.
 * [PERSON_UPPER_BODY_FRACTION] places the aim about a fifth of the way down the box, on the head and shoulders.
 */
fun aimPoint(label: String, left: Float, top: Float, right: Float, bottom: Float): Pair<Float, Float> {
  val aimX = (left + right) / 2f
  val height = (bottom - top).coerceAtLeast(0f)
  val aimY =
    if (label.equals("person", ignoreCase = true)) top + height * PERSON_UPPER_BODY_FRACTION
    else (top + bottom) / 2f
  return aimX to aimY
}

fun tiltToCenter(
  centerX: Float,
  centerY: Float,
  horizontalFovDeg: Float = TiltAdvice.HORIZONTAL_FOV_DEG,
  verticalFovDeg: Float = TiltAdvice.VERTICAL_FOV_DEG,
): TiltAdvice =
  TiltAdvice(
    yawDeg = (centerX - 0.5f) * horizontalFovDeg,
    pitchDeg = (centerY - 0.5f) * verticalFovDeg,
  )
