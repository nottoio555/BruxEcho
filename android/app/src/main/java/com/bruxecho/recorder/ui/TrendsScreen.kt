package com.bruxecho.recorder.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.domain.DemoModeState
import com.bruxecho.recorder.domain.DetectionRepositoryProvider
import com.bruxecho.recorder.domain.NightSummary
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.AccentDim
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import com.bruxecho.recorder.util.formatThaiDate
import java.time.ZoneId

@Composable
fun TrendsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val repository = remember { DetectionRepositoryProvider.get(context) }
    var rangeDays by remember { mutableStateOf(14) }
    var nights by remember { mutableStateOf<List<NightSummary>>(emptyList()) }
    var loading by remember { mutableStateOf(true) }

    LaunchedEffect(rangeDays, DemoModeState.isEnabled) {
        loading = true
        // getRecentNights returns newest-first; the chart reads left (oldest) to right (newest).
        nights = repository.getRecentNights(rangeDays).sortedBy { it.date }
        loading = false
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = Spacing.sm, vertical = Spacing.sm),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.miccheck_back))
            }
            Text(
                text = stringResource(R.string.trends_title),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
        }

        Row(
            modifier = Modifier.padding(horizontal = Spacing.lg),
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            RangeChip(label = stringResource(R.string.trends_range_14), selected = rangeDays == 14) { rangeDays = 14 }
            RangeChip(label = stringResource(R.string.trends_range_30), selected = rangeDays == 30) { rangeDays = 30 }
        }

        Spacer(modifier = Modifier.height(Spacing.lg))

        when {
            loading -> Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = Accent)
            }
            nights.isEmpty() -> Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(text = stringResource(R.string.trends_empty), color = TextSecondary)
            }
            else -> TrendsContent(nights)
        }
    }
}

@Composable
private fun RangeChip(label: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = { Text(text = label) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = Accent,
            selectedLabelColor = OnAccent,
            containerColor = AccentDim,
            labelColor = TextSecondary,
        ),
    )
}

@Composable
private fun TrendsContent(nights: List<NightSummary>) {
    val counts = nights.map { it.grindEventCount }
    val average = counts.average()
    val maxNight = nights.maxByOrNull { it.grindEventCount }
    val minNight = nights.minByOrNull { it.grindEventCount }

    Column(modifier = Modifier.padding(horizontal = Spacing.lg)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xl)) {
            SummaryStat(label = stringResource(R.string.trends_average), value = "%.1f".format(average))
            maxNight?.let {
                SummaryStat(
                    label = stringResource(R.string.trends_max_night),
                    value = "${it.grindEventCount}",
                    caption = formatThaiDate(it.date.atStartOfDay(ZoneId.systemDefault()).toInstant()),
                )
            }
            minNight?.let {
                SummaryStat(
                    label = stringResource(R.string.trends_min_night),
                    value = "${it.grindEventCount}",
                    caption = formatThaiDate(it.date.atStartOfDay(ZoneId.systemDefault()).toInstant()),
                )
            }
        }

        Spacer(modifier = Modifier.height(Spacing.xl))
        BarChart(nights = nights, average = average, modifier = Modifier.fillMaxWidth().height(180.dp))
        Spacer(modifier = Modifier.height(Spacing.sm))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(text = formatThaiDate(nights.first().date.atStartOfDay(ZoneId.systemDefault()).toInstant()), color = TextSecondary, fontSize = 11.sp)
            Text(text = formatThaiDate(nights.last().date.atStartOfDay(ZoneId.systemDefault()).toInstant()), color = TextSecondary, fontSize = 11.sp)
        }
    }
}

@Composable
private fun SummaryStat(label: String, value: String, caption: String? = null) {
    Column {
        Text(text = value, style = tabularNumberStyle(fontSize = 24.sp), color = Accent)
        Text(text = label, color = TextSecondary, fontSize = 12.sp)
        caption?.let { Text(text = it, color = TextSecondary, fontSize = 10.sp) }
    }
}

@Composable
private fun BarChart(nights: List<NightSummary>, average: Double, modifier: Modifier = Modifier) {
    val barColor = Accent
    val zeroBarColor = Accent.copy(alpha = 0.25f)
    val averageLineColor = TextSecondary
    val maxValue = (nights.maxOfOrNull { it.grindEventCount } ?: 0).coerceAtLeast(1)

    Canvas(modifier = modifier) {
        val barCount = nights.size
        val gapPx = 3.dp.toPx()
        val totalGap = gapPx * (barCount - 1).coerceAtLeast(0)
        val barWidth = ((size.width - totalGap) / barCount).coerceAtLeast(1f)
        val chartHeight = size.height

        val avgY = chartHeight - (average / maxValue).toFloat().coerceIn(0f, 1f) * chartHeight
        drawLine(
            color = averageLineColor.copy(alpha = 0.6f),
            start = Offset(0f, avgY),
            end = Offset(size.width, avgY),
            strokeWidth = 1.dp.toPx(),
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f)),
        )

        nights.forEachIndexed { index, night ->
            val fraction = night.grindEventCount / maxValue.toFloat()
            val barHeight = (fraction * chartHeight).coerceAtLeast(2f)
            val x = index * (barWidth + gapPx)
            drawRoundRect(
                color = if (night.grindEventCount == 0) zeroBarColor else barColor,
                topLeft = Offset(x, chartHeight - barHeight),
                size = Size(barWidth, barHeight),
                cornerRadius = CornerRadius(3.dp.toPx(), 3.dp.toPx()),
            )
        }
    }
}
