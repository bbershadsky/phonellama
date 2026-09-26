/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.google.ai.edge.gallery.ui.theme.GalleryTheme

/** Rear-camera tracker. Launch with adb: am start -n com.phonellama.app/com.google.ai.edge.gallery.rover.RoverVisionActivity */
class RoverVisionActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent {
      GalleryTheme {
        RoverVisionScreen(onClose = { finish() })
      }
    }
  }
}
