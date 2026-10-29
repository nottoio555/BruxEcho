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
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import com.bruxecho.recorder.data.SensitivityConfig
import com.bruxecho.recorder.data.SensitivityLevel
import com.bruxecho.recorder.data.SensitivitySettings
import com.bruxecho.recorder.domain.DemoModeState
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.AccentDim
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.util.LOW_SPACE_WARNING_BYTES
import com.bruxecho.recorder.util.StorageInfo
import com.bruxecho.recorder.util.formatBytes
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import kotlinx.coroutines.launch

private sealed class MicCheckUiState {
    data object Idle : MicCheckUiState()
    data class Running(val phase: MicCheckPhase) : MicCheckUiState()
    data class Done(val result: MicCheckResult) : MicCheckUiState()
    data object Failed : MicCheckUiState()
}

// A full 44.1kHz/16-bit/mono night's worth of WAV, used only to forecast
// storage — the actual session may run shorter or longer, and the real sample
// rate is only known once recording begins. Lived in HomeScreen until the
// storage readout moved here.
private const val ESTIMATED_NIGHT_HOURS = 8.0
private const val ESTIMATED_BYTES_PER_NIGHT = (44_100L * 2L * ESTIMATED_NIGHT_HOURS * 3600.0).toLong()

@Composable
fun MicCheckScreen(onBack: (() -> Unit)? = null, onOpenReferenceClips: (() -> Unit)? = null) {
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
            if (onBack != null) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.miccheck_back))
                }
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
        StorageSection()
        Spacer(modifier = Modifier.height(24.dp))

        if (onOpenReferenceClips != null) {
            ReferenceClipsSection(onOpen = onOpenReferenceClips)
            Spacer(modifier = Modifier.height(24.dp))
        }

        DemoModeSection()

        Spacer(modifier = Modifier.height(36.dp))
        SensitivitySection()
    }
}

@Composable
private fun SensitivitySection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var config by remember { mutableStateOf(SensitivityConfig(SensitivityLevel.MEDIUM, SensitivityLevel.MEDIUM.k, SensitivityLevel.MEDIUM.bandRatio)) }

    LaunchedEffect(Unit) {
        SensitivitySettings.observe(context).collect { config = it }
    }

    Column {
        Text(
            text = stringResource(R.string.sensitivity_title),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = stringResource(R.string.sensitivity_body),
            color = TextSecondary,
            fontSize = 12.sp,
        )
        Spacer(modifier = Modifier.height(12.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SensitivityChip(stringResource(R.string.sensitivity_low), config.level == SensitivityLevel.LOW) {
                scope.launch { SensitivitySettings.setLevel(context, SensitivityLevel.LOW) }
            }
            SensitivityChip(stringResource(R.string.sensitivity_medium), config.level == SensitivityLevel.MEDIUM) {
                scope.launch { SensitivitySettings.setLevel(context, SensitivityLevel.MEDIUM) }
            }
            SensitivityChip(stringResource(R.string.sensitivity_high), config.level == SensitivityLevel.HIGH) {
                scope.launch { SensitivitySettings.setLevel(context, SensitivityLevel.HIGH) }
            }
            SensitivityChip(stringResource(R.string.sensitivity_custom), config.level == SensitivityLevel.CUSTOM) {
                scope.launch { SensitivitySettings.setCustom(context, config.k, config.bandRatio) }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = stringResource(R.string.sensitivity_actual_values, config.k, config.bandRatio),
            style = tabularNumberStyle(fontSize = 13.sp),
            color = TextSecondary,
        )

        if (config.level == SensitivityLevel.CUSTOM) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(text = stringResource(R.string.sensitivity_k_label, config.k), color = MaterialTheme.colorScheme.onBackground, fontSize = 13.sp)
            Slider(
                value = config.k,
                onValueChange = { config = config.copy(k = it) },
                onValueChangeFinished = { scope.launch { SensitivitySettings.setCustom(context, config.k, config.bandRatio) } },
                valueRange = 2.0f..10.0f,
                colors = SliderDefaults.colors(thumbColor = Accent, activeTrackColor = Accent),
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(R.string.sensitivity_band_ratio_label, config.bandRatio), color = MaterialTheme.colorScheme.onBackground, fontSize = 13.sp)
            Slider(
                value = config.bandRatio,
                onValueChange = { config = config.copy(bandRatio = it) },
                onValueChangeFinished = { scope.launch { SensitivitySettings.setCustom(context, config.k, config.bandRatio) } },
                valueRange = 0.0f..1.0f,
                colors = SliderDefaults.colors(thumbColor = Accent, activeTrackColor = Accent),
            )
        }
    }
}

@Composable
private fun SensitivityChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(text = label, fontSize = 12.sp) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = Accent,
            selectedLabelColor = OnAccent,
            containerColor = AccentDim,
            labelColor = TextSecondary,
        ),
    )
}

@Composable
private fun StorageSection() {
    val context = LocalContext.current
    val nightsCount = remember { StorageInfo.completedNightsCount(context) }
    val freeBytes = remember { StorageInfo.freeBytes(context) }
    val estimatedRemainingNights = (freeBytes / ESTIMATED_BYTES_PER_NIGHT).toInt()

    Column {
        Text(
            text = stringResource(R.string.settings_storage_title),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.home_nights_recorded, nightsCount),
            color = TextSecondary,
            fontSize = 13.sp,
        )
        Text(
            text = stringResource(R.string.home_free_space, formatBytes(freeBytes)),
            color = TextSecondary,
            fontSize = 13.sp,
        )
        Text(
            text = stringResource(
                R.string.home_storage_forecast,
                formatBytes(ESTIMATED_BYTES_PER_NIGHT),
                estimatedRemainingNights,
            ),
            color = TextSecondary,
            fontSize = 13.sp,
        )
        if (freeBytes < LOW_SPACE_WARNING_BYTES) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(R.string.home_low_space_warning), color = Warn, fontSize = 13.sp)
        }
        // No battery-exemption prompt here: this screen already carries one
        // further up, with its own buttons. Moving Home's copy down here
        // verbatim put the same warning on the page twice.
    }
}

@Composable
private fun ReferenceClipsSection(onOpen: () -> Unit) {
    Column {
        Text(
            text = stringResource(R.string.settings_reference_clips),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = stringResource(R.string.settings_reference_clips_body),
                color = TextSecondary,
                fontSize = 13.sp,
                modifier = Modifier.weight(1f),
            )
            Spacer(modifier = Modifier.width(12.dp))
            OutlinedButton(onClick = onOpen) {
                Text(text = stringResource(R.string.reference_clips_title), fontSize = 13.sp)
            }
        }
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
