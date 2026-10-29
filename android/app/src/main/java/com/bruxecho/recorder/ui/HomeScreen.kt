package com.bruxecho.recorder.ui

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.BatteryManager
import android.provider.Settings
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.LifestyleTagEntity
import com.bruxecho.recorder.service.RecordingState
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface2
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import com.bruxecho.recorder.util.StorageInfo
import com.bruxecho.recorder.util.formatBytes
import com.bruxecho.recorder.util.formatHms
import com.bruxecho.recorder.util.isIgnoringBatteryOptimizations
import java.time.LocalDate
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private val RECORD_BUTTON_SIZE = 140.dp
private const val NIGHT_MODE_IDLE_TIMEOUT_MS = 30_000L
private const val DEEP_DIM_BRIGHTNESS = 0.01f


@Composable
fun HomeScreen(
    recordingState: RecordingState.Snapshot,
    micPermissionPermanentlyDenied: Boolean,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenSessions: () -> Unit,
    onOpenResults: () -> Unit,
    onOpenAppSettings: () -> Unit,
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopBar(onOpenSettings = onOpenSettings, onOpenSessions = onOpenSessions, onOpenResults = onOpenResults)
        Box(modifier = Modifier.fillMaxSize().padding(Spacing.xl)) {
            if (recordingState.isRecording) {
                RecordingBody(recordingState = recordingState, onStopRecording = onStopRecording)
            } else {
                IdleBody(
                    onStartRecording = onStartRecording,
                    errorMessage = recordingState.lastError,
                    micPermissionPermanentlyDenied = micPermissionPermanentlyDenied,
                    onOpenAppSettings = onOpenAppSettings,
                )
            }
        }
    }
}

@Composable
private fun TopBar(onOpenSettings: () -> Unit, onOpenSessions: () -> Unit, onOpenResults: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = Spacing.lg, vertical = Spacing.lg),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Image(
                painter = painterResource(R.drawable.logo_wordmark),
                contentDescription = stringResource(R.string.app_name),
                modifier = Modifier.height(28.dp),
            )
        }
        Row {
            IconButton(onClick = onOpenResults) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ShowChart,
                    contentDescription = stringResource(R.string.results_icon_description),
                    tint = TextSecondary,
                )
            }
            IconButton(onClick = onOpenSessions) {
                Icon(
                    imageVector = Icons.Filled.History,
                    contentDescription = stringResource(R.string.sessions_icon_description),
                    tint = TextSecondary,
                )
            }
            IconButton(onClick = onOpenSettings) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = stringResource(R.string.settings_icon_description),
                    tint = TextSecondary,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun IdleBody(
    onStartRecording: () -> Unit,
    errorMessage: String?,
    micPermissionPermanentlyDenied: Boolean,
    onOpenAppSettings: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Button(
            onClick = onStartRecording,
            modifier = Modifier.size(RECORD_BUTTON_SIZE),
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = Accent, contentColor = OnAccent),
        ) {
            Icon(imageVector = Icons.Filled.Mic, contentDescription = null, modifier = Modifier.size(56.dp))
        }
        Spacer(modifier = Modifier.height(Spacing.xl))
        Text(
            text = stringResource(R.string.home_start_button),
            fontSize = 18.sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onBackground,
        )
        if (errorMessage != null) {
            Spacer(modifier = Modifier.height(Spacing.md))
            if (micPermissionPermanentlyDenied) {
                Text(
                    text = stringResource(R.string.permission_denied_permanently_body),
                    color = ErrorRed,
                    fontSize = 13.sp,
                    modifier = Modifier.padding(horizontal = Spacing.xl),
                )
                Spacer(modifier = Modifier.height(Spacing.sm))
                OutlinedButton(onClick = onOpenAppSettings) {
                    Text(text = stringResource(R.string.permission_denied_open_settings))
                }
            } else {
                Text(text = errorMessage, color = ErrorRed, fontSize = 13.sp)
            }
        }
    }

}

@Composable
private fun RecordingBody(recordingState: RecordingState.Snapshot, onStopRecording: () -> Unit) {
    val context = LocalContext.current
    var showConfirmStop by remember { mutableStateOf(false) }
    var isNightMode by remember { mutableStateOf(false) }
    var interactionTick by remember { mutableIntStateOf(0) }

    LaunchedEffect(interactionTick) {
        isNightMode = false
        delay(NIGHT_MODE_IDLE_TIMEOUT_MS)
        isNightMode = true
    }

    // A deeper dim on top of the always-on recording dim (see MainActivity)
    // once the screen has been ignored for a while — restored the instant
    // the user taps anywhere, same as leaving night mode.
    LaunchedEffect(isNightMode) {
        val activity = context as? Activity ?: return@LaunchedEffect
        val params = activity.window.attributes
        params.screenBrightness = if (isNightMode) DEEP_DIM_BRIGHTNESS else RECORDING_DIM_BRIGHTNESS
        activity.window.attributes = params
    }

    fun registerInteraction() {
        interactionTick++
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures(onTap = { registerInteraction() })
            },
        contentAlignment = Alignment.Center,
    ) {
        if (isNightMode) {
            NightModeView(
                elapsedTotalS = recordingState.elapsedTotalS,
                onStopClick = {
                    registerInteraction()
                    showConfirmStop = true
                },
            )
        } else {
            FullRecordingView(
                recordingState = recordingState,
                onStopClick = {
                    registerInteraction()
                    showConfirmStop = true
                },
            )
        }
    }

    if (showConfirmStop) {
        AlertDialog(
            onDismissRequest = { showConfirmStop = false },
            title = { Text(text = stringResource(R.string.recording_confirm_stop_title)) },
            text = { Text(text = stringResource(R.string.recording_confirm_stop_body)) },
            confirmButton = {
                Button(
                    onClick = {
                        showConfirmStop = false
                        onStopRecording()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed, contentColor = Color.White),
                ) {
                    Text(text = stringResource(R.string.recording_confirm_stop_yes))
                }
            },
            dismissButton = {
                TextButton(onClick = { showConfirmStop = false }) {
                    Text(text = stringResource(R.string.recording_confirm_stop_cancel))
                }
            },
        )
    }
}

@Composable
private fun FullRecordingView(recordingState: RecordingState.Snapshot, onStopClick: () -> Unit) {
    val isCharging = rememberIsCharging()

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = formatHms(recordingState.elapsedTotalS),
            style = tabularNumberStyle(fontSize = 56.sp),
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(Spacing.xl))
        LevelMeter(rms = recordingState.currentRms)
        Spacer(modifier = Modifier.height(Spacing.lg))
        Text(
            text = stringResource(
                R.string.recording_status_format,
                "${recordingState.sampleRateActual / 1000.0} kHz",
                recordingState.sourceLabel,
                formatBytes(recordingState.bytesWrittenTotal),
            ),
            color = TextSecondary,
            fontSize = 13.sp,
        )

        if (!isCharging) {
            Spacer(modifier = Modifier.height(Spacing.md))
            Text(text = stringResource(R.string.home_charging_warning), color = Warn, fontSize = 12.sp)
        }

        Spacer(modifier = Modifier.height(Spacing.xl))
        Text(text = stringResource(R.string.recording_privacy_banner), color = Accent, fontSize = 12.sp)
        Spacer(modifier = Modifier.height(Spacing.xxl))
        Button(
            onClick = onStopClick,
            colors = ButtonDefaults.buttonColors(containerColor = ErrorRed, contentColor = Color.White),
            shape = RoundedCornerShape(50),
        ) {
            Icon(imageVector = Icons.Filled.Stop, contentDescription = null)
            Spacer(modifier = Modifier.width(Spacing.sm))
            Text(text = stringResource(R.string.recording_stop_button), fontSize = 16.sp)
        }
    }
}

@Composable
private fun NightModeView(elapsedTotalS: Double, onStopClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = formatHms(elapsedTotalS),
            style = tabularNumberStyle(fontSize = 48.sp),
            color = TextSecondary,
        )
        Spacer(modifier = Modifier.height(Spacing.xl))
        OutlinedButton(onClick = onStopClick) {
            Icon(imageVector = Icons.Filled.Stop, contentDescription = null, tint = TextSecondary)
            Spacer(modifier = Modifier.width(Spacing.sm))
            Text(text = stringResource(R.string.recording_stop_button), color = TextSecondary)
        }
        Spacer(modifier = Modifier.height(Spacing.xxl))
        Text(text = stringResource(R.string.recording_tap_to_wake), color = TextSecondary.copy(alpha = 0.5f), fontSize = 11.sp)
    }
}

@Composable
private fun rememberIsCharging(): Boolean {
    val context = LocalContext.current
    var charging by remember { mutableStateOf(false) }
    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(receiverContext: Context, intent: Intent) {
                val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
            }
        }
        val sticky = context.registerReceiver(receiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        sticky?.let { receiver.onReceive(context, it) }
        onDispose { context.unregisterReceiver(receiver) }
    }
    return charging
}

@Composable
private fun LevelMeter(rms: Float) {
    val displayLevel = (rms * 6f).coerceIn(0f, 1f)
    val animated by animateFloatAsState(targetValue = displayLevel, label = "rms")

    Box(
        modifier = Modifier
            .fillMaxWidth(0.8f)
            .height(14.dp)
            .background(Surface2, RoundedCornerShape(7.dp)),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth(animated.coerceIn(0.02f, 1f))
                .height(14.dp)
                .background(if (animated > 0.85f) Warn else Accent, RoundedCornerShape(7.dp)),
        )
    }
}
