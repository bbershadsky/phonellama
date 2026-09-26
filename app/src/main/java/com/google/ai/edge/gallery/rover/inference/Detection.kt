/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.inference

/** One detector hit. Box edges are normalized to the upright frame, 0..1. */
data class Detection(
  val label: String,
  val score: Float,
  val left: Float,
  val top: Float,
  val right: Float,
  val bottom: Float,
) {
  val centerX: Float
    get() = (left + right) / 2f

  val centerY: Float
    get() = (top + bottom) / 2f

  val width: Float
    get() = (right - left).coerceAtLeast(0f)

  val height: Float
    get() = (bottom - top).coerceAtLeast(0f)

  val area: Float
    get() = width * height

  val isPerson: Boolean
    get() = label.equals(PERSON_LABEL, ignoreCase = true)

  companion object {
    const val PERSON_LABEL = "person"
  }
}

fun List<Detection>.onlyPersons(): List<Detection> = filter { it.isPerson }
