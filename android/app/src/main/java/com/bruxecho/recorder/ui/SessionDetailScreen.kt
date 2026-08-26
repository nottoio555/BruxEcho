package com.bruxecho.recorder.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.session.InterruptionMeta
import com.bruxecho.recorder.session.ParsedSession
import com.bruxecho.recorder.session.SegmentMeta
import com.bruxecho.recorder.session.SessionReader
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.util.StorageInfo
import com.bruxecho.recorder.util.formatBytes
import com.bruxecho.recorder.util.formatDurationThai
import com.bruxecho.recorder.util.formatThaiDate
import com.bruxecho.recorder.util.formatTimeOnly
import java.io.File

@Composable
fun SessionDetailScreen(sessionId: String, onBack: () -> Unit, onDeleted: () -> Unit, onOpenResults: (String) -> Unit) {
    val context = LocalContext.current
    val dir = remember(sessionId) { File(StorageInfo.sessionsRootDir(context), sessionId) }
    val parsed = remember(sessionId) { SessionReader.readOne(dir) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

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
            Column {
                Text(
                    text = stringResource(R.string.session_detail_title),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground,
                )
                Text(text = sessionId, color = TextSecondary, fontSize = 13.sp)
            }
        }
        Spacer(modifier = Modifier.height(16.dp))

        if (parsed == null) {
            Text(text = stringResource(R.string.session_detail_unreadable), color = ErrorRed)
        } else {
            SessionFields(parsed)
            Spacer(modifier = Modifier.height(20.dp))
            SegmentsSection(dir = dir, segments = parsed.segments)
            Spacer(modifier = Modifier.height(20.dp))
            InterruptionsSection(interruptions = parsed.interruptions)
            Spacer(modifier = Modifier.height(28.dp))
            HorizontalDivider(color = TextSecondary.copy(alpha = 0.2f))
            Spacer(modifier = Modifier.height(16.dp))
            if (parsed.endedAt != null) {
                Button(onClick = { onOpenResults(sessionId) }) {
                    Text(text = stringResource(R.string.session_detail_view_results))
                }
            } else {
                // No detection model exists yet either way (Sprint 6), but an
                // incomplete recording has no usable audio to analyze at all.
                Text(
                    text = stringResource(R.string.session_detail_future_section),
                    color = TextSecondary,
                    fontSize = 12.sp,
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
        OutlinedButton(
            onClick = { showDeleteConfirm = true },
            colors = ButtonDefaults.outlinedButtonColors(contentColor = ErrorRed),
        ) {
            Text(text = stringResource(R.string.session_detail_delete_button))
        }
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text(text = stringResource(R.string.session_detail_delete_confirm_title)) },
            text = { Text(text = stringResource(R.string.session_detail_delete_confirm_body)) },
            confirmButton = {
                Button(
                    onClick = {
                        SessionReader.deleteSession(dir)
                        showDeleteConfirm = false
                        onDeleted()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed, contentColor = Color.White),
                ) {
                    Text(text = stringResource(R.string.session_detail_delete_confirm_yes))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text(text = stringResource(R.string.session_detail_delete_confirm_cancel))
                }
            },
        )
    }
}

@Composable
private fun SessionFields(parsed: ParsedSession) {
    val onBg = MaterialTheme.colorScheme.onBackground

    Text(text = formatThaiDate(parsed.startedAt), fontWeight = FontWeight.Medium, fontSize = 16.sp, color = onBg)
    Text(
        text = "${formatTimeOnly(parsed.startedAt)} - ${formatTimeOnly(parsed.endedAt)} · ${formatDurationThai(parsed.totalDurationS)}",
        color = TextSecondary,
        fontSize = 14.sp,
    )
    Spacer(modifier = Modifier.height(12.dp))

    FieldLine(stringResource(R.string.session_detail_device, parsed.deviceModel ?: "?", parsed.androidVersion ?: "?"))
    FieldLine(stringResource(R.string.session_detail_source, parsed.audioSourceUsed ?: "?"))
    parsed.audioSourceFallbackReason?.let {
        FieldLine(stringResource(R.string.session_detail_source_fallback, it))
    }
    FieldLine(
        stringResource(
            R.string.session_detail_sample_rate,
            parsed.sampleRateRequested ?: 0,
            parsed.sampleRateActual ?: 0,
        ),
    )
    FieldLine(stringResource(R.string.session_detail_format, parsed.channels ?: 0, parsed.bitDepth ?: 0))
    FieldLine(
        stringResource(
            R.string.session_detail_effects_disabled,
            parsed.effectsDisabled.ifEmpty { listOf(stringResource(R.string.session_detail_none)) }.joinToString(", "),
        ),
    )
    if (parsed.effectsUnavailable.isNotEmpty()) {
        FieldLine(stringResource(R.string.session_detail_effects_unavailable, parsed.effectsUnavailable.joinToString(", ")))
    }
}

@Composable
private fun FieldLine(text: String) {
    Text(text = text, color = TextSecondary, fontSize = 13.sp, modifier = Modifier.padding(vertical = 2.dp))
}

@Composable
private fun SegmentsSection(dir: File, segments: List<SegmentMeta>) {
    Text(
        text = stringResource(R.string.session_detail_segments_title, segments.size),
        fontWeight = FontWeight.Medium,
        fontSize = 15.sp,
        color = MaterialTheme.colorScheme.onBackground,
    )
    Spacer(modifier = Modifier.height(8.dp))
    segments.forEach { seg ->
        val sizeBytes = remember(seg.file) { File(dir, seg.file).length() }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(text = seg.file, color = TextSecondary, fontSize = 13.sp)
            Text(text = "${formatDurationThai(seg.durationS)} · ${formatBytes(sizeBytes)}", color = TextSecondary, fontSize = 13.sp)
        }
        Spacer(modifier = Modifier.height(4.dp))
    }
}

@Composable
private fun InterruptionsSection(interruptions: List<InterruptionMeta>) {
    Text(
        text = stringResource(R.string.session_detail_interruptions_title, interruptions.size),
        fontWeight = FontWeight.Medium,
        fontSize = 15.sp,
        color = MaterialTheme.colorScheme.onBackground,
    )
    Spacer(modifier = Modifier.height(8.dp))
    if (interruptions.isEmpty()) {
        Text(text = stringResource(R.string.session_detail_no_interruptions), color = TextSecondary, fontSize = 13.sp)
        return
    }
    interruptions.forEach { gap ->
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(text = "${formatTimeOnly(gap.at)} · ${interruptionReasonLabel(gap.reason)}", color = TextSecondary, fontSize = 13.sp)
            Text(text = formatDurationThai(gap.gapS), color = TextSecondary, fontSize = 13.sp)
        }
        Spacer(modifier = Modifier.height(4.dp))
    }
}

@Composable
private fun interruptionReasonLabel(reason: String): String = when (reason) {
    "audio_focus_lost" -> stringResource(R.string.interruption_reason_audio_focus_lost)
    else -> reason
}
