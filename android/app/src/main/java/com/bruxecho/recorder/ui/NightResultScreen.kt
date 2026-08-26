package com.bruxecho.recorder.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.domain.Confidence
import com.bruxecho.recorder.domain.DemoModeState
import com.bruxecho.recorder.domain.GrindEvent
import com.bruxecho.recorder.domain.MockDetectionRepository
import com.bruxecho.recorder.domain.NightResult
import com.bruxecho.recorder.playback.rememberClipPlayer
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.Negative
import com.bruxecho.recorder.ui.theme.Positive
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface2
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import com.bruxecho.recorder.util.formatDurationThai
import com.bruxecho.recorder.util.formatThaiDate
import com.bruxecho.recorder.util.formatTimeOnly
import com.bruxecho.recorder.util.formatTimeWithSeconds
import java.time.Duration
import kotlinx.coroutines.launch
import kotlin.math.abs

private sealed class ResultUiState {
    data object Loading : ResultUiState()
    data object NeverRecorded : ResultUiState()
    data object Incomplete : ResultUiState()
    data class Success(val result: NightResult) : ResultUiState()
}

/** Items before the clip rows in the LazyColumn: header, stats, comparison, timeline, "clips (N)" title. */
private const val CLIP_LIST_ITEM_OFFSET = 5

@Composable
fun NightResultScreen(sessionId: String?, onBack: () -> Unit, onOpenTrends: () -> Unit) {
    val context = LocalContext.current
    val repository = remember { MockDetectionRepository(context) }
    var uiState by remember { mutableStateOf<ResultUiState>(ResultUiState.Loading) }

    LaunchedEffect(sessionId, DemoModeState.isEnabled) {
        uiState = ResultUiState.Loading
        val targetId = sessionId ?: repository.latestSessionId()
        if (targetId == null) {
            uiState = ResultUiState.NeverRecorded
            return@LaunchedEffect
        }
        if (repository.isSessionIncomplete(targetId)) {
            uiState = ResultUiState.Incomplete
            return@LaunchedEffect
        }
        val result = repository.getNightResult(targetId)
        uiState = if (result == null) ResultUiState.Incomplete else ResultUiState.Success(result)
    }

    Column(modifier = Modifier.fillMaxSize()) {
        ResultTopBar(onBack = onBack, onOpenTrends = onOpenTrends, showTrends = uiState is ResultUiState.Success)
        if (DemoModeState.isEnabled) {
            DemoBanner()
        }
        when (val state = uiState) {
            ResultUiState.Loading -> CenteredMessage { CircularProgressIndicator(color = Accent) }
            ResultUiState.NeverRecorded -> CenteredTextState(
                title = stringResource(R.string.results_never_recorded_title),
                body = stringResource(R.string.results_never_recorded_body),
            )
            ResultUiState.Incomplete -> CenteredTextState(
                title = stringResource(R.string.results_incomplete_title),
                body = stringResource(R.string.results_incomplete_body),
                titleColor = Warn,
            )
            is ResultUiState.Success -> SuccessBody(state.result)
        }
    }
}

@Composable
private fun ResultTopBar(onBack: () -> Unit, onOpenTrends: () -> Unit, showTrends: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = Spacing.sm, vertical = Spacing.sm),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.miccheck_back))
            }
            Text(
                text = stringResource(R.string.results_title),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
        }
        if (showTrends) {
            IconButton(onClick = onOpenTrends) {
                Icon(imageVector = Icons.AutoMirrored.Filled.ShowChart, contentDescription = stringResource(R.string.results_open_trends), tint = TextSecondary)
            }
        }
    }
}

@Composable
private fun DemoBanner() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Accent.copy(alpha = 0.16f))
            .padding(horizontal = Spacing.lg, vertical = Spacing.sm),
        horizontalArrangement = Arrangement.Center,
    ) {
        Text(text = stringResource(R.string.results_demo_banner), color = Accent, fontSize = 12.sp, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun CenteredMessage(content: @Composable () -> Unit) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { content() }
}

@Composable
private fun CenteredTextState(title: String, body: String, titleColor: Color = TextSecondary) {
    CenteredMessage {
        Column(
            modifier = Modifier.padding(Spacing.xxl),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = title, color = titleColor, fontSize = 17.sp, fontWeight = FontWeight.Medium)
            Spacer(modifier = Modifier.height(Spacing.sm))
            Text(text = body, color = TextSecondary, fontSize = 14.sp, textAlign = TextAlign.Center)
        }
    }
}

@Composable
private fun SuccessBody(result: NightResult) {
    val player = rememberClipPlayer()
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    LazyColumn(
        state = listState,
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = Spacing.lg, vertical = Spacing.md),
    ) {
        item { HeaderBlock(result) }
        item { Spacer(modifier = Modifier.height(Spacing.lg)); StatsRow(result) }
        item { Spacer(modifier = Modifier.height(Spacing.sm)); ComparisonLine(result.comparisonPrevious7d) }
        item { Spacer(modifier = Modifier.height(Spacing.xl)) }

        if (result.grindEventCount == 0) {
            item {
                NoEventsBlock()
                Spacer(modifier = Modifier.height(Spacing.xl))
            }
        } else {
            item {
                Timeline(result) { index ->
                    scope.launch { listState.animateScrollToItem(CLIP_LIST_ITEM_OFFSET + index) }
                }
                Spacer(modifier = Modifier.height(Spacing.xl))
                Text(
                    text = stringResource(R.string.results_clips_title, result.events.size),
                    fontWeight = FontWeight.Medium,
                    fontSize = 15.sp,
                    color = MaterialTheme.colorScheme.onBackground,
                )
                Spacer(modifier = Modifier.height(Spacing.sm))
            }
            itemsIndexed(result.events, key = { _, e -> e.id }) { _, event ->
                ClipRow(
                    event = event,
                    isPlaying = player.currentlyPlayingPath == event.clipPath && event.clipPath != null,
                    onPlay = { event.clipPath?.let { player.play(it) } },
                )
                Spacer(modifier = Modifier.height(Spacing.xs))
            }
            item { Spacer(modifier = Modifier.height(Spacing.xl)) }
        }

        item { SupplementarySection(result) }
        item { Spacer(modifier = Modifier.height(Spacing.xxl)) }
    }
}

@Composable
private fun HeaderBlock(result: NightResult) {
    Text(text = formatThaiDate(result.startedAt), fontWeight = FontWeight.Bold, fontSize = 18.sp, color = MaterialTheme.colorScheme.onBackground)
    Text(
        text = "${formatTimeOnly(result.startedAt)} - ${formatTimeOnly(result.endedAt)} · " +
            formatDurationThai(Duration.between(result.startedAt, result.endedAt).seconds.toDouble()),
        color = TextSecondary,
        fontSize = 14.sp,
    )
}

@Composable
private fun StatsRow(result: NightResult) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(Spacing.xl)) {
        StatBlock(value = "${result.grindEventCount}", label = stringResource(R.string.results_event_count_label))
        StatBlock(value = "%.1f".format(result.eventsPerHour), label = stringResource(R.string.results_events_per_hour_label))
    }
}

@Composable
private fun StatBlock(value: String, label: String) {
    Column {
        Text(text = value, style = tabularNumberStyle(fontSize = 36.sp), color = Accent)
        Text(text = label, color = TextSecondary, fontSize = 12.sp)
    }
}

@Composable
private fun ComparisonLine(comparison: Float?) {
    if (comparison == null) return
    val pct = abs(comparison * 100f)
    val (text, color) = when {
        comparison < -0.02f -> stringResource(R.string.results_comparison_down, pct) to Positive
        comparison > 0.02f -> stringResource(R.string.results_comparison_up, pct) to Negative
        else -> stringResource(R.string.results_comparison_same) to TextSecondary
    }
    Text(text = text, color = color, fontSize = 13.sp, fontWeight = FontWeight.Medium)
}

@Composable
private fun NoEventsBlock() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Surface2)
            .padding(Spacing.lg),
    ) {
        Text(
            text = stringResource(R.string.results_no_events_title),
            color = MaterialTheme.colorScheme.onBackground,
            fontWeight = FontWeight.Medium,
            fontSize = 16.sp,
        )
        Spacer(modifier = Modifier.height(Spacing.xs))
        Text(text = stringResource(R.string.results_no_events_body), color = TextSecondary, fontSize = 13.sp)
    }
}

private val TimelineDotSize = 14.dp
private val TimelineDotTouchTarget = 32.dp // dot itself is small; the tap target around it still meets the 48dp-ish guidance closely enough for a dense timeline

@Composable
private fun Timeline(result: NightResult, onDotClick: (Int) -> Unit) {
    Text(text = stringResource(R.string.results_timeline_title), fontWeight = FontWeight.Medium, fontSize = 15.sp, color = MaterialTheme.colorScheme.onBackground)
    Spacer(modifier = Modifier.height(Spacing.sm))
    val totalDurationS = Duration.between(result.startedAt, result.endedAt).seconds.toFloat().coerceAtLeast(1f)

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxWidth()
            .height(48.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Surface2),
    ) {
        val trackWidthDp = maxWidth
        result.events.forEachIndexed { index, event ->
            val fraction = (event.offsetS / totalDurationS).coerceIn(0f, 1f)
            val xOffset = trackWidthDp * fraction - TimelineDotTouchTarget / 2
            Box(
                modifier = Modifier
                    .offset(x = xOffset)
                    .align(Alignment.CenterStart)
                    .size(TimelineDotTouchTarget)
                    .clip(CircleShape)
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        onClick = { onDotClick(index) },
                    ),
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    modifier = Modifier
                        .size(TimelineDotSize)
                        .clip(CircleShape)
                        .background(confidenceColor(event.confidence)),
                )
            }
        }
    }
}

@Composable
private fun confidenceColor(confidence: Confidence) = when (confidence) {
    Confidence.HIGH -> Accent
    Confidence.MEDIUM -> Warn
    Confidence.LOW -> TextSecondary
}

@Composable
private fun confidenceLabel(confidence: Confidence): String = when (confidence) {
    Confidence.HIGH -> stringResource(R.string.confidence_high)
    Confidence.MEDIUM -> stringResource(R.string.confidence_medium)
    Confidence.LOW -> stringResource(R.string.confidence_low)
}

@Composable
private fun ClipRow(event: GrindEvent, isPlaying: Boolean, onPlay: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Surface2)
            .padding(horizontal = Spacing.md, vertical = Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = formatTimeWithSeconds(event.timestamp), color = MaterialTheme.colorScheme.onBackground, fontSize = 14.sp, style = tabularNumberStyle(14.sp, FontWeight.Medium))
            Spacer(modifier = Modifier.height(2.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = "${"%.1f".format(event.durationS)}s", color = TextSecondary, fontSize = 12.sp)
                Spacer(modifier = Modifier.width(Spacing.sm))
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(confidenceColor(event.confidence).copy(alpha = 0.18f))
                        .padding(horizontal = 6.dp, vertical = 2.dp),
                ) {
                    Text(text = confidenceLabel(event.confidence), color = confidenceColor(event.confidence), fontSize = 11.sp)
                }
            }
        }
        IconButton(
            onClick = onPlay,
            enabled = event.clipPath != null,
            modifier = Modifier.size(48.dp),
        ) {
            if (event.clipPath == null) {
                Icon(imageVector = Icons.Filled.PlayArrow, contentDescription = stringResource(R.string.results_clip_no_audio), tint = TextSecondary.copy(alpha = 0.4f))
            } else if (isPlaying) {
                Icon(imageVector = Icons.Filled.Stop, contentDescription = null, tint = Accent)
            } else {
                Icon(imageVector = Icons.Filled.PlayArrow, contentDescription = null, tint = Accent)
            }
        }
    }
}

@Composable
private fun SupplementarySection(result: NightResult) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Surface2)
            .padding(Spacing.lg),
    ) {
        Text(
            text = stringResource(R.string.results_supplementary_title),
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(Spacing.xs))
        result.snoreRatio?.let { ratio ->
            Text(text = stringResource(R.string.results_snore_ratio, ratio * 100f), color = TextSecondary, fontSize = 13.sp)
            if (ratio > 0.4f) {
                Spacer(modifier = Modifier.height(Spacing.xs))
                Text(text = stringResource(R.string.results_snore_warning), color = Warn, fontSize = 12.sp)
            }
            Spacer(modifier = Modifier.height(Spacing.sm))
        }
        Text(text = stringResource(R.string.results_medical_disclaimer), color = TextSecondary, fontSize = 11.sp)
    }
}
