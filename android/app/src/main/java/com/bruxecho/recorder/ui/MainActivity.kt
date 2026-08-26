package com.bruxecho.recorder.ui

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.bruxecho.recorder.R
import com.bruxecho.recorder.service.RecordingService
import com.bruxecho.recorder.service.RecordingState
import com.bruxecho.recorder.ui.theme.BruxEchoTheme

/** Screen brightness while recording — dim, but not off, so the elapsed time is still glanceable. */
const val RECORDING_DIM_BRIGHTNESS = 0.03f

private sealed class Screen {
    data object Home : Screen()
    data object MicCheck : Screen()
    data object SessionsList : Screen()
    data class SessionDetail(val sessionId: String) : Screen()
    data class Results(val sessionId: String?) : Screen()
    data object Trends : Screen()
}

class MainActivity : ComponentActivity() {

    private var onPermissionResult: ((Boolean) -> Unit)? = null
    private var permissionPermanentlyDenied by mutableStateOf(false)

    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { results ->
        val granted = results[Manifest.permission.RECORD_AUDIO] == true
        if (!granted) {
            // By the time this callback fires we've already shown the system
            // prompt at least once, so "rationale no longer needed" reliably
            // means the user picked "don't ask again" / it's now permanent —
            // the standard Android idiom for detecting that state.
            permissionPermanentlyDenied = !shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)
        }
        onPermissionResult?.invoke(granted)
        onPermissionResult = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        setContent {
            BruxEchoTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    var screen by remember { mutableStateOf<Screen>(Screen.Home) }
                    val recordingState by RecordingState.state.collectAsState()

                    LaunchedEffect(recordingState.isRecording) {
                        setScreenDim(recordingState.isRecording)
                    }

                    when (val current = screen) {
                        Screen.Home -> HomeScreen(
                            recordingState = recordingState,
                            micPermissionPermanentlyDenied = permissionPermanentlyDenied,
                            onStartRecording = { requestAudioPermissionThenStart() },
                            onStopRecording = { RecordingService.stop(this@MainActivity) },
                            onOpenSettings = { screen = Screen.MicCheck },
                            onOpenSessions = { screen = Screen.SessionsList },
                            onOpenResults = { screen = Screen.Results(sessionId = null) },
                            onOpenAppSettings = { openAppSettings() },
                        )
                        Screen.MicCheck -> MicCheckScreen(
                            onBack = { screen = Screen.Home },
                        )
                        Screen.SessionsList -> SessionsListScreen(
                            onBack = { screen = Screen.Home },
                            onOpenSession = { sessionId -> screen = Screen.SessionDetail(sessionId) },
                        )
                        is Screen.SessionDetail -> SessionDetailScreen(
                            sessionId = current.sessionId,
                            onBack = { screen = Screen.SessionsList },
                            onDeleted = { screen = Screen.SessionsList },
                            onOpenResults = { sessionId -> screen = Screen.Results(sessionId) },
                        )
                        is Screen.Results -> NightResultScreen(
                            sessionId = current.sessionId,
                            onBack = { screen = Screen.Home },
                            onOpenTrends = { screen = Screen.Trends },
                        )
                        Screen.Trends -> TrendsScreen(
                            onBack = { screen = Screen.Results(sessionId = null) },
                        )
                    }
                }
            }
        }
    }

    private fun requestAudioPermissionThenStart() {
        val needed = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            needed += Manifest.permission.POST_NOTIFICATIONS
        }
        val missing = needed.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            permissionPermanentlyDenied = false
            RecordingState.update { it.copy(lastError = null) }
            RecordingService.start(this)
            return
        }
        onPermissionResult = { granted ->
            if (granted) {
                permissionPermanentlyDenied = false
                RecordingState.update { it.copy(lastError = null) }
                RecordingService.start(this)
            } else {
                RecordingState.update { it.copy(lastError = getString(R.string.permission_denied_message)) }
            }
        }
        requestPermissionsLauncher.launch(missing.toTypedArray())
    }

    private fun openAppSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName"))
        startActivity(intent)
    }

    private fun setScreenDim(dim: Boolean) {
        val activity: Activity = this
        val params = activity.window.attributes
        params.screenBrightness = if (dim) RECORDING_DIM_BRIGHTNESS else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        activity.window.attributes = params
    }
}
