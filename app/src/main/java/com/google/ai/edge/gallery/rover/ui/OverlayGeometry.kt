/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.ui

data class ContentRect(
  val left: Float,
  val top: Float,
  val width: Float,
  val height: Float,
)

/** Letterboxed rectangle of an upright frame drawn with fit-center. */
fun contentRect(viewWidth: Float, viewHeight: Float, imageWidth: Float, imageHeight: Float): ContentRect {
  if (viewWidth <= 0f || viewHeight <= 0f || imageWidth <= 0f || imageHeight <= 0f) {
    return ContentRect(0f, 0f, viewWidth.coerceAtLeast(0f), viewHeight.coerceAtLeast(0f))
  }
  val imageAspect = imageWidth / imageHeight
  val viewAspect = viewWidth / viewHeight
  return if (imageAspect > viewAspect) {
    val height = viewWidth / imageAspect
    ContentRect(0f, (viewHeight - height) / 2f, viewWidth, height)
  } else {
    val width = viewHeight * imageAspect
    ContentRect((viewWidth - width) / 2f, 0f, width, viewHeight)
  }
}

fun viewPointToNormalized(
  xPx: Float,
  yPx: Float,
  rect: ContentRect,
): Pair<Float, Float>? {
  if (rect.width <= 0f || rect.height <= 0f) return null
  val x = (xPx - rect.left) / rect.width
  val y = (yPx - rect.top) / rect.height
  if (x !in 0f..1f || y !in 0f..1f) return null
  return x to y
}
