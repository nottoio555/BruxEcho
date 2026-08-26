package com.bruxecho.recorder.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.audio.MicCheckPhase
import com.bruxecho.recorder.audio.MicCheckRecorder
import com.bruxecho.recorder.audio.MicCheckResult
import com.bruxecho.recorder.domain.DemoModeState
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import kotlinx.coroutines.launch

private sealed class MicCheckUiState {
    data object Idle : MicCheckUiState()
    data class Running(val phase: MicCheckPhase) : MicCheckUiState()
    data class Done(val result: MicCheckResult) : MicCheckUiState()
    data object Failed : MicCheckUiState()
}

@Composable
fun MicCheckScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var uiState by remember { mutableStateOf<MicCheckUiState>(MicCheckUiState.Idle) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.miccheck_back))
            }
            Text(
                text = stringResource(R.string.miccheck_title),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
        }

        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = stringResource(R.string.miccheck_intro),
            color = TextSecondary,
            fontSize = 14.sp,
        )
        Spacer(modifier = Modifier.height(20.dp))

        when (val state = uiState) {
            is MicCheckUiState.Idle -> {
                Button(
                    onClick = {
                        uiState = MicCheckUiState.Running(MicCheckPhase.RECORDING)
                        scope.launch {
                            val result = MicCheckRecorder.runCheck(
                                onPhase = { phase -> uiState = MicCheckUiState.Running(phase) },
                            )
                            uiState = if (result != null) MicCheckUiState.Done(result) else MicCheckUiState.Failed
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Accent, contentColor = OnAccent),
                ) {
                    Text(text = stringResource(R.string.miccheck_start_button))
                }
            }
            is MicCheckUiState.Running -> {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(color = Accent, modifier = Modifier.height(20.dp))
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = stringResource(
                            if (state.phase == MicCheckPhase.RECORDING) R.string.miccheck_recording else R.string.miccheck_analyzing,
                        ),
                        color = TextSecondary,
                    )
                }
            }
            is MicCheckUiState.Done -> {
                MicCheckResultView(state.result)
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedButton(onClick = { uiState = MicCheckUiState.Idle }) {
                    Text(text = stringResource(R.string.miccheck_start_button))
                }
            }
            is MicCheckUiState.Failed -> {
                Text(text = stringResource(R.string.miccheck_failed), color = ErrorRed)
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedButton(onClick = { uiState = MicCheckUiState.Idle }) {
                    Text(text = stringResource(R.string.miccheck_start_button))
                }
            }
        }

        Spacer(modifier = Modifier.height(36.dp))
        BatterySection(onRequestExemption = {
            val powerManager = context.getSystemService(PowerManager::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
                powerManager?.isIgnoringBatteryOptimizations(context.packageName) != true
            ) {
                val intent = Intent(
                    Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:${context.packageName}"),
                )
                context.startActivity(intent)
            }
        }, onOpenAppSettings = {
            val intent = Intent(
                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.parse("package:${context.packageName}"),
            )
            context.startActivity(intent)
        })

        Spacer(modifier = Modifier.height(36.dp))
        DemoModeSection()
    }
}

@Composable
private fun DemoModeSection() {
    val context = LocalContext.current
    Column {
        Text(
            text = stringResource(R.string.settings_demo_mode_title),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = stringResource(R.string.settings_demo_mode_body),
                color = TextSecondary,
                fontSize = 13.sp,
                modifier = Modifier.weight(1f),
            )
            Spacer(modifier = Modifier.width(12.dp))
            Switch(
                checked = DemoModeState.isEnabled,
                onCheckedChange = { DemoModeState.setEnabled(context, it) },
                colors = SwitchDefaults.colors(checkedTrackColor = Accent, checkedThumbColor = OnAccent),
            )
        }
    }
}

@Composable
private fun MicCheckResultView(result: MicCheckResult) {
    Column {
        Text(
            text = stringResource(R.string.miccheck_result_source, result.sourceLabel),
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = stringResource(
                R.string.miccheck_result_sample_rate,
                result.sampleRateRequested,
                result.sampleRateActual,
            ),
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = stringResource(R.string.miccheck_result_band_energy, result.bandEnergyPercent),
            color = MaterialTheme.colorScheme.onBackground,
        )
        result.fallbackReason?.let {
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = it, color = Warn, fontSize = 13.sp)
        }
        Spacer(modifier = Modifier.height(14.dp))
        Text(
            text = stringResource(
                if (result.dspLikelyStillFiltering) R.string.miccheck_warning_dsp else R.string.miccheck_ok,
            ),
            color = if (result.dspLikelyStillFiltering) ErrorRed else Accent,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
private fun BatterySection(onRequestExemption: () -> Unit, onOpenAppSettings: () -> Unit) {
    Column {
        Text(
            text = stringResource(R.string.battery_section_title),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.battery_section_body),
            color = TextSecondary,
            fontSize = 13.sp,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedButton(onClick = onRequestExemption) {
                Text(text = stringResource(R.string.battery_request_exemption), fontSize = 12.sp)
            }
            OutlinedButton(onClick = onOpenAppSettings) {
                Text(text = stringResource(R.string.battery_open_app_settings), fontSize = 12.sp)
            }
        }
    }
}
