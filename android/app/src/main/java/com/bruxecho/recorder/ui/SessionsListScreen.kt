package com.bruxecho.recorder.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.service.RecordingState
import com.bruxecho.recorder.session.SessionListEntry
import com.bruxecho.recorder.session.SessionReader
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.util.StorageInfo
import com.bruxecho.recorder.util.formatBytes
import com.bruxecho.recorder.util.formatDurationThai
import com.bruxecho.recorder.util.formatThaiDate
import com.bruxecho.recorder.util.formatTimeOnly

@Composable
fun SessionsListScreen(onBack: () -> Unit, onOpenSession: (String) -> Unit) {
    val context = LocalContext.current
    val activeSessionId = RecordingState.state.value.sessionId
    val entries = remember { SessionReader.listSessions(StorageInfo.sessionsRootDir(context)) }
    val freeBytes = remember { StorageInfo.freeBytes(context) }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.miccheck_back))
            }
            Text(
                text = stringResource(R.string.sessions_title),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
        }
        Text(
            text = stringResource(R.string.sessions_free_space, formatBytes(freeBytes)),
            color = TextSecondary,
            fontSize = 13.sp,
            modifier = Modifier.padding(start = 20.dp, bottom = 12.dp),
        )

        if (entries.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(text = stringResource(R.string.sessions_empty), color = TextSecondary)
            }
            return@Column
        }

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp),
        ) {
            items(entries, key = { it.sessionId }) { entry ->
                SessionListItem(
                    entry = entry,
                    isActive = entry.sessionId == activeSessionId,
                    onClick = { onOpenSession(entry.sessionId) },
                )
                Spacer(modifier = Modifier.height(10.dp))
            }
        }
    }
}

@Composable
private fun SessionListItem(entry: SessionListEntry, isActive: Boolean, onClick: () -> Unit) {
    val parsed = entry.parsed

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Surface1)
            .clickable(onClick = onClick)
            .padding(16.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = formatThaiDate(parsed?.startedAt),
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onBackground,
                fontSize = 15.sp,
            )
            when {
                isActive -> Text(text = stringResource(R.string.sessions_recording_now), color = Accent, fontSize = 12.sp)
                // A crashed/killed recording usually still has a session.json (written
                // throughout, not just at the end) — it's just missing ended_at, not
                // absent entirely. Both cases mean "no usable result", so both get the badge.
                parsed == null || parsed.endedAt == null -> IncompleteBadge()
                parsed.interruptions.isNotEmpty() -> InterruptionBadge(count = parsed.interruptions.size)
            }
        }

        if (parsed != null) {
            Spacer(modifier = Modifier.height(4.dp))
            val endedLabel = if (isActive) "…" else formatTimeOnly(parsed.endedAt)
            Text(
                text = "${formatTimeOnly(parsed.startedAt)} - $endedLabel · ${formatDurationThai(parsed.totalDurationS)}",
                color = TextSecondary,
                fontSize = 13.sp,
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "${parsed.audioSourceUsed ?: "?"} · ${parsed.segments.size} segment · ${formatBytes(entry.totalWavBytes)}",
                color = TextSecondary,
                fontSize = 12.sp,
            )
        } else {
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = entry.sessionId, color = TextSecondary, fontSize = 13.sp)
        }
    }
}

@Composable
private fun IncompleteBadge() {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(imageVector = Icons.Filled.Warning, contentDescription = null, tint = ErrorRed, modifier = Modifier.height(16.dp))
        Spacer(modifier = Modifier.width(4.dp))
        Text(text = stringResource(R.string.sessions_incomplete), color = ErrorRed, fontSize = 12.sp)
    }
}

@Composable
private fun InterruptionBadge(count: Int) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(imageVector = Icons.Filled.Warning, contentDescription = null, tint = Warn, modifier = Modifier.height(16.dp))
        Spacer(modifier = Modifier.width(4.dp))
        Text(text = stringResource(R.string.sessions_interruptions_count, count), color = Warn, fontSize = 12.sp)
    }
}
