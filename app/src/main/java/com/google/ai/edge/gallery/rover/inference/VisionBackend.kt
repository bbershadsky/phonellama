/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover.inference

import android.graphics.Bitmap
import java.io.Closeable

/**
 * Fast object detector used by the tracking loop.
 *
 * Implementations must return quickly enough for a 15 Hz or faster camera loop.
 * A vision-language model belongs on a separate event-driven path and must not
 * implement this interface.
 */
interface VisionBackend : Closeable {
  val name: String

  suspend fun detect(frame: ImageFrame): List<Detection>
}

/** One upright camera frame. The caller recycles [bitmap] after [VisionBackend.detect]. */
class ImageFrame(
  val bitmap: Bitmap,
  val receivedNs: Long,
)
