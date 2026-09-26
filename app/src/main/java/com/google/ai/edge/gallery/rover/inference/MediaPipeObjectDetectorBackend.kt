/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.inference

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector

/** Bundled COCO detectors. All three use the same MediaPipe object-detector task. */
enum class DetectorModel(val label: String, val shortLabel: String, val assetPath: String) {
  SSD("SSD MobileNet", "SSD", "models/ssd_mobilenet_v2.tflite"),
  LITE0("EfficientDet Lite0", "Lite0", "models/efficientdet_lite0.tflite"),
  LITE2("EfficientDet Lite2", "Lite2", "models/efficientdet_lite2.tflite"),
}

/**
 * MediaPipe Tasks object detector. COCO labels include "person".
 * The model file is the bundled asset for [model].
 */
class MediaPipeObjectDetectorBackend private constructor(
  private val detector: ObjectDetector,
  val model: DetectorModel,
) : VisionBackend {
  override val name: String = model.label

  override suspend fun detect(frame: ImageFrame): List<Detection> {
    val bitmap = softwareBitmap(frame.bitmap)
    val mpImage = BitmapImageBuilder(bitmap).build()
    val result = detector.detect(mpImage)
    val width = bitmap.width.toFloat().coerceAtLeast(1f)
    val height = bitmap.height.toFloat().coerceAtLeast(1f)
    if (bitmap !== frame.bitmap) bitmap.recycle()
    return result.detections().mapNotNull { hit ->
      val category = hit.categories().firstOrNull() ?: return@mapNotNull null
      val score = category.score()
      if (score < MIN_SCORE) return@mapNotNull null
      val box = hit.boundingBox()
      Detection(
        label = category.categoryName().ifBlank { "object" },
        score = score,
        left = (box.left / width).coerceIn(0f, 1f),
        top = (box.top / height).coerceIn(0f, 1f),
        right = (box.right / width).coerceIn(0f, 1f),
        bottom = (box.bottom / height).coerceIn(0f, 1f),
      )
    }
  }

  override fun close() {
    detector.close()
  }

  companion object {
    const val MIN_SCORE = 0.6f

    fun create(context: Context, model: DetectorModel = DetectorModel.LITE0): VisionBackend {
      return try {
        val base =
          BaseOptions.builder()
            .setModelAssetPath(model.assetPath)
            .build()
        val options =
          ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setMaxResults(8)
            .setScoreThreshold(MIN_SCORE)
            .build()
        MediaPipeObjectDetectorBackend(ObjectDetector.createFromOptions(context, options), model)
      } catch (error: Exception) {
        UnavailableVisionBackend(error.message ?: "Object detector failed to load")
      }
    }
  }
}

class UnavailableVisionBackend(val reason: String) : VisionBackend {
  override val name: String = "unavailable"

  override suspend fun detect(frame: ImageFrame): List<Detection> = emptyList()

  override fun close() = Unit
}

private fun softwareBitmap(bitmap: Bitmap): Bitmap {
  if (bitmap.config != Bitmap.Config.HARDWARE) return bitmap
  return bitmap.copy(Bitmap.Config.ARGB_8888, false)
}
