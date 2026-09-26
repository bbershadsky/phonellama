/*
 * Copyright 2026 PhoneLlama
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.rover

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.ai.edge.gallery.rover.benchmark.dropPercent
import com.google.ai.edge.gallery.rover.inference.DetectorModel
import com.google.ai.edge.gallery.rover.inference.MediaPipeObjectDetectorBackend
import com.google.ai.edge.gallery.rover.safety.StopReason
import com.google.ai.edge.gallery.rover.tracking.TargetSelector
import com.google.ai.edge.gallery.rover.tracking.TrackStrategy
import com.google.ai.edge.gallery.rover.tracking.Track
import com.google.ai.edge.gallery.rover.ui.contentRect
import com.google.ai.edge.gallery.rover.ui.aimPoint
import com.google.ai.edge.gallery.rover.ui.tiltToCenter
import com.google.ai.edge.gallery.rover.ui.viewPointToNormalized
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

@Composable
fun RoverVisionScreen(onClose: () -> Unit) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  var permissionGranted by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
    )
  }
  val permissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      permissionGranted = granted
    }
  var session by remember { mutableStateOf<RoverSession?>(null) }
  var cameraError by remember { mutableStateOf<String?>(null) }
  val sessionRef = remember { AtomicReference<RoverSession?>(null) }
  val providerRef = remember { AtomicReference<ProcessCameraProvider?>(null) }
  val previewRef = remember { AtomicReference<PreviewView?>(null) }
  val useFrontRef = remember { AtomicReference(false) }
  val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
  var useFront by remember { mutableStateOf(false) }
  var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }

  LaunchedEffect(Unit) {
    if (!permissionGranted) permissionLauncher.launch(Manifest.permission.CAMERA)
    val backend =
      withContext(Dispatchers.Default) { MediaPipeObjectDetectorBackend.create(context.applicationContext) }
    val created = RoverSession(backend, context.applicationContext)
    sessionRef.set(created)
    session = created
  }

  LaunchedEffect(Unit) {
    while (true) {
      delay(100)
      nowMs = System.currentTimeMillis()
    }
  }

  DisposableEffect(session) {
    val current = session
    if (current != null) sessionRef.set(current)
    onDispose {
      if (sessionRef.get() === current) sessionRef.set(null)
      current?.close()
    }
  }

  DisposableEffect(Unit) {
    onDispose {
      providerRef.get()?.unbindAll()
      analysisExecutor.shutdown()
    }
  }

  val ui by
    produceState(initialValue = RoverUiState(status = "Starting detector"), session) {
      val active = session ?: return@produceState
      active.state.collect { value = it }
    }

  Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
    if (permissionGranted) {
      AndroidView(
        factory = { ctx ->
          PreviewView(ctx).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            previewRef.set(this)
            post {
              val providerFuture = ProcessCameraProvider.getInstance(ctx)
              providerFuture.addListener(
                {
                  val provider = providerFuture.get()
                  providerRef.set(provider)
                  bindCamera(
                    provider = provider,
                    previewView = this,
                    lifecycleOwner = lifecycleOwner,
                    executor = analysisExecutor,
                    sessionRef = sessionRef,
                    useFront = useFrontRef.get(),
                    onError = { cameraError = it },
                  )
                },
                ContextCompat.getMainExecutor(ctx),
              )
            }
          }
        },
        modifier = Modifier.fillMaxSize(),
      )
      DetectionOverlay(
        ui = ui,
        onSelect = { id -> sessionRef.get()?.selectTrack(id) },
        modifier = Modifier.fillMaxSize(),
      )
    }

    Column(
      modifier =
        Modifier.align(Alignment.BottomCenter)
          .fillMaxWidth()
          .background(Color.Black.copy(alpha = 0.72f))
          .navigationBarsPadding()
          .padding(12.dp),
      verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
      val status = cameraError?.takeIf { it.isNotBlank() } ?: ui.status
      Text(
        status,
        color = Color.White,
        fontSize = 12.sp,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
      )
      StrategyDropdown(selected = ui.strategy, onSelect = { sessionRef.get()?.setStrategy(it) })
      HudTable(ui = ui, nowMs = nowMs)
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        Button(onClick = { sessionRef.get()?.clearTarget() }, modifier = Modifier.weight(1f)) {
          Text("Reset", maxLines = 1)
        }
        DetectorDropdown(
          selected = ui.detector,
          onSelect = { sessionRef.get()?.setDetector(it) },
          modifier = Modifier.weight(1f),
        )
      }
    }

    Row(
      modifier = Modifier.align(Alignment.TopStart).statusBarsPadding(),
      verticalAlignment = Alignment.CenterVertically,
    ) {
      TextButton(onClick = onClose) { Text("Close", color = Color.White) }
      TextButton(onClick = { sessionRef.get()?.setPersonsOnly(!ui.personsOnly) }) {
        Text(
          if (ui.personsOnly) "People" else "All",
          color = if (ui.personsOnly) Color(0xFF69F0AE) else Color.White,
        )
      }
      TextButton(
        onClick = {
          val nextFront = !useFrontRef.get()
          useFrontRef.set(nextFront)
          useFront = nextFront
          sessionRef.get()?.setFacingFront(nextFront)
          val provider = providerRef.get()
          val preview = previewRef.get()
          if (provider != null && preview != null) {
            bindCamera(
              provider = provider,
              previewView = preview,
              lifecycleOwner = lifecycleOwner,
              executor = analysisExecutor,
              sessionRef = sessionRef,
              useFront = nextFront,
              onError = { cameraError = it },
            )
          }
        }
      ) {
        Text(if (useFront) "Rear" else "Front", color = Color.White)
      }
    }

    if (!permissionGranted) {
      Column(
        modifier = Modifier.align(Alignment.Center).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        Text("Camera permission is required for object tracking.", color = Color.White)
        Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Grant camera") }
      }
    }
  }
}

@Composable
private fun DetectionOverlay(
  ui: RoverUiState,
  onSelect: (Int) -> Unit,
  modifier: Modifier = Modifier,
) {
  val textMeasurer = rememberTextMeasurer()
  var viewSize by remember { mutableStateOf(IntSize.Zero) }
  Canvas(
    modifier =
      modifier.onSizeChanged { viewSize = it }.pointerInput(ui.tracks, ui.frameWidth, ui.frameHeight, viewSize) {
        detectTapGestures { offset ->
          val rect =
            contentRect(
              viewSize.width.toFloat(),
              viewSize.height.toFloat(),
              ui.frameWidth.toFloat(),
              ui.frameHeight.toFloat(),
            )
          val point = viewPointToNormalized(offset.x, offset.y, rect) ?: return@detectTapGestures
          val track = TargetSelector.trackAt(ui.tracks, point.first, point.second) ?: return@detectTapGestures
          onSelect(track.id)
        }
      }
  ) {
    val rect = contentRect(size.width, size.height, ui.frameWidth.toFloat(), ui.frameHeight.toFloat())
    for (track in ui.tracks) {
      val locked = track.id == ui.lockedTrackId
      val color =
        when {
          locked && track.lostFrames > 0 -> Color(0xFFFFB74D)
          locked -> Color(0xFF69F0AE)
          else -> Color.White
        }
      val left = rect.left + track.left * rect.width
      val top = rect.top + track.top * rect.height
      val boxWidth = track.width * rect.width
      val boxHeight = track.height * rect.height
      drawRect(
        color = color,
        topLeft = Offset(left, top),
        size = ComposeSize(boxWidth, boxHeight),
        style = Stroke(width = if (locked) 6f else 3f),
      )
      if (locked) {
        val caption = "${track.label.take(7)} ${(track.score * 100).toInt()}"
        val layout = textMeasurer.measure(caption, style = TextStyle(color = color, fontSize = 11.sp))
        val textWidth = layout.size.width.toFloat()
        val textLeft = left.coerceIn(rect.left, (rect.left + rect.width - textWidth).coerceAtLeast(rect.left))
        val textTop =
          if (top - layout.size.height >= rect.top) top - layout.size.height else (top + 2f).coerceAtMost(rect.top + rect.height - layout.size.height)
        drawText(textMeasurer, caption, Offset(textLeft, textTop), TextStyle(color = color, fontSize = 11.sp))
      }
    }
    val lockedTrack = ui.tracks.find { it.id == ui.lockedTrackId }
    if (lockedTrack != null) drawTiltArrow(rect, lockedTrack, textMeasurer)
  }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTiltArrow(
  rect: com.google.ai.edge.gallery.rover.ui.ContentRect,
  track: Track,
  textMeasurer: androidx.compose.ui.text.TextMeasurer,
) {
  val (aimX, aimY) = aimPoint(track.label, track.left, track.top, track.right, track.bottom)
  val advice = tiltToCenter(aimX, aimY)
  val origin = Offset(rect.left + rect.width / 2f, rect.top + rect.height / 2f)
  val aim = Offset(rect.left + aimX * rect.width, rect.top + aimY * rect.height)
  val color = if (track.lostFrames > 0) Color(0xFFFFB74D) else Color(0xFF40C4FF)
  drawCircle(color, radius = 6f, center = aim)
  if (advice.centered) {
    drawCircle(color, radius = 10f, center = origin, style = Stroke(width = 3f))
    return
  }
  val maxLength = minOf(rect.width, rect.height) * 0.22f
  val reach = (advice.magnitudeDeg / 25f).coerceIn(0.35f, 1f)
  val length = maxLength * reach
  val scale = length / advice.magnitudeDeg.coerceAtLeast(0.01f)
  val tip = Offset(origin.x + advice.yawDeg * scale, origin.y + advice.pitchDeg * scale)
  drawLine(color, origin, tip, strokeWidth = 8f, cap = StrokeCap.Round)
  val angle = atan2(tip.y - origin.y, tip.x - origin.x)
  val head = 22f
  val wing = 0.45f
  drawLine(
    color,
    tip,
    Offset(tip.x + head * cos(angle + PI.toFloat() - wing), tip.y + head * sin(angle + PI.toFloat() - wing)),
    strokeWidth = 8f,
    cap = StrokeCap.Round,
  )
  drawLine(
    color,
    tip,
    Offset(tip.x + head * cos(angle + PI.toFloat() + wing), tip.y + head * sin(angle + PI.toFloat() + wing)),
    strokeWidth = 8f,
    cap = StrokeCap.Round,
  )
  val label = "${advice.magnitudeDeg.format0()}°"
  val layout = textMeasurer.measure(label, style = TextStyle(color = color, fontSize = 12.sp))
  val labelLeft = (tip.x + 8f).coerceIn(rect.left, (rect.left + rect.width - layout.size.width).coerceAtLeast(rect.left))
  val labelTop = (tip.y - layout.size.height - 4f).coerceIn(rect.top, (rect.top + rect.height - layout.size.height).coerceAtLeast(rect.top))
  drawText(textMeasurer, label, Offset(labelLeft, labelTop), TextStyle(color = color, fontSize = 12.sp))
}

private fun bindCamera(
  provider: ProcessCameraProvider,
  previewView: PreviewView,
  lifecycleOwner: androidx.lifecycle.LifecycleOwner,
  executor: java.util.concurrent.Executor,
  sessionRef: AtomicReference<RoverSession?>,
  useFront: Boolean,
  onError: (String?) -> Unit,
) {
  val selector = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
  if (!provider.hasCamera(selector)) {
    onError(if (useFront) "This phone has no front camera." else "This phone has no rear camera.")
    return
  }
  val resolutionSelector =
    ResolutionSelector.Builder()
      .setResolutionStrategy(
        ResolutionStrategy(
          Size(640, 480),
          ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
        )
      )
      .build()
  val preview = Preview.Builder().setResolutionSelector(resolutionSelector).build()
  val analysis =
    ImageAnalysis.Builder()
      .setResolutionSelector(resolutionSelector)
      .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
      .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
      .build()
  val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
  preview.targetRotation = rotation
  analysis.targetRotation = rotation
  analysis.setAnalyzer(executor) { image ->
    try {
      val receivedNs = System.nanoTime()
      val bitmap = uprightBitmap(image, mirrorHorizontally = useFront)
      val active = sessionRef.get()
      if (active == null) bitmap.recycle() else active.offer(bitmap, receivedNs)
    } catch (error: Exception) {
      Log.e("RoverVision", "camera frame failed", error)
    } finally {
      image.close()
    }
  }
  preview.surfaceProvider = previewView.surfaceProvider
  try {
    provider.unbindAll()
    provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
    onError(null)
  } catch (error: Exception) {
    Log.e("RoverVision", "camera bind failed", error)
    onError(error.message ?: "Camera failed to start")
  }
}

private fun uprightBitmap(image: ImageProxy, mirrorHorizontally: Boolean): Bitmap {
  val source = image.toBitmap()
  val matrix = Matrix()
  val rotation = image.imageInfo.rotationDegrees
  if (rotation != 0) matrix.postRotate(rotation.toFloat())
  if (mirrorHorizontally) matrix.postScale(-1f, 1f)
  val upright =
    if (matrix.isIdentity) source
    else Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
  if (upright !== source) source.recycle()
  if (upright.config != Bitmap.Config.HARDWARE) return upright
  val copy = upright.copy(Bitmap.Config.ARGB_8888, false)
  upright.recycle()
  return copy
}

@Composable
private fun DetectorDropdown(
  selected: DetectorModel,
  onSelect: (DetectorModel) -> Unit,
  modifier: Modifier = Modifier,
) {
  var open by remember { mutableStateOf(false) }
  Box(modifier = modifier) {
    Button(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) {
      Text(selected.shortLabel, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
    DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
      DetectorModel.entries.forEach { model ->
        DropdownMenuItem(
          text = { Text(model.label, maxLines = 1) },
          onClick = {
            onSelect(model)
            open = false
          },
        )
      }
    }
  }
}

@Composable
private fun StrategyDropdown(selected: TrackStrategy, onSelect: (TrackStrategy) -> Unit) {
  var open by remember { mutableStateOf(false) }
  Box(modifier = Modifier.fillMaxWidth()) {
    TextButton(onClick = { open = true }) {
      Text(
        "Track: ${selected.label}",
        color = Color.White,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        fontSize = 13.sp,
      )
    }
    DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
      TrackStrategy.entries.forEach { strategy ->
        DropdownMenuItem(
          text = { Text(strategy.label, maxLines = 1) },
          onClick = {
            onSelect(strategy)
            open = false
          },
        )
      }
    }
  }
}

@Composable
private fun HudTable(ui: RoverUiState, nowMs: Long) {
  val command = ui.command
  val expired = command != null && nowMs - command.timestampMs > command.ttlMs
  val linear = if (command == null || expired) 0f else command.linearMps
  val angular = if (command == null || expired) 0f else command.angularRps
  val locked = ui.tracks.find { it.id == ui.lockedTrackId }
  val tilt =
    locked?.let {
      val (aimX, aimY) = aimPoint(it.label, it.left, it.top, it.right, it.bottom)
      tiltToCenter(aimX, aimY)
    }
  val rows =
    listOf(
      listOf("fps", "drop", "inf", "e2e"),
      listOf(
        ui.fps.format1(),
        "${dropPercent(ui.droppedFrames, ui.processedFrames).format0()}%",
        "${ui.inferenceP50Ms.format0()}/${ui.inferenceP95Ms.format0()}",
        "${ui.e2eP50Ms.format0()}/${ui.e2eP95Ms.format0()}",
      ),
      listOf("lock", "state", "v", "w"),
      listOf(
        ui.lockedTrackId?.let { "#$it" } ?: "—",
        if (expired) "ttl" else shortReason(ui.reason),
        linear.format2(),
        angular.format2(),
      ),
      listOf("tilt", "yaw", "pitch", "mm"),
      listOf(
        tilt?.let { "${it.magnitudeDeg.format0()}°" } ?: "—",
        tilt?.let { signedDeg(it.yawDeg, "R", "L") } ?: "—",
        tilt?.let { signedDeg(it.pitchDeg, "D", "U") } ?: "—",
        ui.frontDistanceMm.toString(),
      ),
    )
  Column(modifier = Modifier.fillMaxWidth()) {
    rows.forEachIndexed { index, cells ->
      val header = index % 2 == 0
      Row(modifier = Modifier.fillMaxWidth()) {
        cells.forEach { cell ->
          Text(
            cell,
            modifier = Modifier.weight(1f),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            color = if (header) Color.White.copy(alpha = 0.65f) else Color.White,
          )
        }
      }
    }
  }
}

private fun shortReason(reason: StopReason): String =
  when (reason) {
    StopReason.NONE -> "ok"
    StopReason.NO_TARGET -> "none"
    StopReason.LOST_TARGET -> "lost"
    StopReason.LOW_CONFIDENCE -> "low"
    StopReason.OBSTACLE -> "block"
    StopReason.SENSOR_FAULT -> "fault"
    StopReason.LOW_BATTERY -> "batt"
    StopReason.ESTOP -> "stop"
  }

private fun signedDeg(value: Float, positive: String, negative: String): String {
  val side = if (value >= 0f) positive else negative
  return "${abs(value).format0()}°$side"
}

private fun Float.format1(): String = String.format("%.1f", this)

private fun Float.format0(): String = String.format("%.0f", this)

private fun Float.format2(): String = String.format("%.2f", this)
