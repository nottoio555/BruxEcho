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
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bedtime
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.LifestyleTagEntity
import java.time.LocalDate
import kotlinx.coroutines.launch
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.bruxecho.recorder.R
import com.bruxecho.recorder.service.RecordingService
import com.bruxecho.recorder.service.RecordingState
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.AccentDim
import com.bruxecho.recorder.ui.theme.BruxEchoTheme
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.TextSecondary

/** Screen brightness while recording — dim, but not off, so the elapsed time is still glanceable. */
const val RECORDING_DIM_BRIGHTNESS = 0.03f

private enum class BottomTab { SLEEP, LOG, SETTINGS }

/**
 * Screens reached by drilling into a specific night/session — rendered
 * full-screen with their own back button, on top of the 3-tab bottom nav
 * rather than as a 4th tab, since they're not top-level destinations.
 */
private sealed class OverlayScreen {
    data object SessionsList : OverlayScreen()
    data class SessionDetail(val sessionId: String) : OverlayScreen()
    data class Results(val sessionId: String?) : OverlayScreen()
    data object Trends : OverlayScreen()
    data object ReferenceClips : OverlayScreen()
    data object LifestyleTags : OverlayScreen()
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
                    var bottomTab by remember { mutableStateOf(BottomTab.SLEEP) }
                    // A real (if tiny) back-stack rather than one nullable screen, so
                    // "back" from e.g. SessionDetail returns to SessionsList when reached
                    // that way, but straight to the Log tab when reached from Calendar —
                    // whichever is actually true for how this instance was opened.
                    var overlayStack by remember { mutableStateOf<List<OverlayScreen>>(emptyList()) }
                    val recordingState by RecordingState.state.collectAsState()

                    fun push(screen: OverlayScreen) {
                        overlayStack = overlayStack + screen
                    }
                    fun pop() {
                        overlayStack = overlayStack.dropLast(1)
                    }

                    LaunchedEffect(recordingState.isRecording) {
                        setScreenDim(recordingState.isRecording)
                    }

                    val currentOverlay = overlayStack.lastOrNull()
                    if (currentOverlay != null) {
                        when (currentOverlay) {
                            OverlayScreen.SessionsList -> SessionsListScreen(
                                onBack = { pop() },
                                onOpenSession = { sessionId -> push(OverlayScreen.SessionDetail(sessionId)) },
                            )
                            is OverlayScreen.SessionDetail -> SessionDetailScreen(
                                sessionId = currentOverlay.sessionId,
                                onBack = { pop() },
                                onDeleted = { pop() },
                                onOpenResults = { sessionId -> push(OverlayScreen.Results(sessionId)) },
                            )
                            is OverlayScreen.Results -> NightResultScreen(
                                sessionId = currentOverlay.sessionId,
                                onBack = { pop() },
                                onOpenTrends = { push(OverlayScreen.Trends) },
                            )
                            OverlayScreen.Trends -> TrendsScreen(
                                onBack = { pop() },
                            )
                            OverlayScreen.ReferenceClips -> ReferenceClipsScreen(
                                onBack = { pop() },
                            )
                            OverlayScreen.LifestyleTags -> LifestyleTagScreen(
                                onDone = { selectedTags, memo ->
                                    pop()
                                    if (selectedTags.isNotEmpty()) {
                                        val today = LocalDate.now()
                                        val dao = BruxEchoDatabase.getInstance(this@MainActivity)
                                            .lifestyleTagDao()
                                        lifecycleScope.launch {
                                            dao.insertAll(
                                                selectedTags.map { tag ->
                                                    LifestyleTagEntity(
                                                        date = today,
                                                        tag = tag,
                                                        memo = memo.ifBlank { null },
                                                    )
                                                },
                                            )
                                        }
                                    }
                                    requestAudioPermissionThenStart()
                                },
                                onCancel = { pop() },
                            )
                        }
                    } else {
                        Scaffold(
                            bottomBar = {
                                BottomTabBar(selected = bottomTab, onSelect = { bottomTab = it })
                            },
                        ) { innerPadding ->
                            Box(modifier = Modifier.padding(innerPadding)) {
                                when (bottomTab) {
                                    BottomTab.SLEEP -> HomeScreen(
                                        recordingState = recordingState,
                                        micPermissionPermanentlyDenied = permissionPermanentlyDenied,
                                        onStartRecording = { push(OverlayScreen.LifestyleTags) },
                                        onStopRecording = { RecordingService.stop(this@MainActivity) },
                                        onOpenSettings = { bottomTab = BottomTab.SETTINGS },
                                        onOpenSessions = { push(OverlayScreen.SessionsList) },
                                        onOpenResults = { push(OverlayScreen.Results(sessionId = null)) },
                                        onOpenAppSettings = { openAppSettings() },
                                    )
                                    BottomTab.LOG -> LogScreen(
                                        onOpenSessionDetail = { sessionId -> push(OverlayScreen.SessionDetail(sessionId)) },
                                        onOpenResults = { sessionId -> push(OverlayScreen.Results(sessionId)) },
                                    )
                                    BottomTab.SETTINGS -> MicCheckScreen(
                                        onBack = null,
                                        onOpenReferenceClips = { push(OverlayScreen.ReferenceClips) },
                                    )
                                }
                            }
                        }
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

@Composable
private fun BottomTabBar(selected: BottomTab, onSelect: (BottomTab) -> Unit) {
    NavigationBar(containerColor = Surface1) {
        NavigationBarItem(
            selected = selected == BottomTab.SLEEP,
            onClick = { onSelect(BottomTab.SLEEP) },
            icon = { Icon(imageVector = Icons.Filled.Bedtime, contentDescription = null) },
            label = { Text(text = stringResource(R.string.bottom_tab_sleep)) },
            colors = bottomTabColors(),
        )
        NavigationBarItem(
            selected = selected == BottomTab.LOG,
            onClick = { onSelect(BottomTab.LOG) },
            icon = { Icon(imageVector = Icons.Filled.CalendarMonth, contentDescription = null) },
            label = { Text(text = stringResource(R.string.bottom_tab_log)) },
            colors = bottomTabColors(),
        )
        NavigationBarItem(
            selected = selected == BottomTab.SETTINGS,
            onClick = { onSelect(BottomTab.SETTINGS) },
            icon = { Icon(imageVector = Icons.Filled.Settings, contentDescription = null) },
            label = { Text(text = stringResource(R.string.bottom_tab_settings)) },
            colors = bottomTabColors(),
        )
    }
}

@Composable
private fun bottomTabColors() = NavigationBarItemDefaults.colors(
    selectedIconColor = Accent,
    selectedTextColor = Accent,
    indicatorColor = AccentDim,
    unselectedIconColor = TextSecondary,
    unselectedTextColor = TextSecondary,
)
