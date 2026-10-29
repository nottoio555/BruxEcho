package com.bruxecho.recorder.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.data.SessionSync
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.DailySummaryEntity
import com.bruxecho.recorder.data.db.LifestyleTag
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.AccentDim
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import com.bruxecho.recorder.util.formatDurationThai
import java.time.LocalDate

private const val MIN_DAYS_PER_GROUP = 5
private const val FACTOR_WINDOW_DAYS = 90L

private enum class StatsRange(val days: Long) { WEEK(7), MONTH(30), YEAR(365) }

private data class FactorResult(val tag: LifestyleTag, val withAvg: Double, val withoutAvg: Double, val withCount: Int, val withoutCount: Int)

@Composable
fun StatsScreen() {
    val context = LocalContext.current
    val db = remember { BruxEchoDatabase.getInstance(context) }
    var range by remember { mutableStateOf(StatsRange.MONTH) }
    var summaries by remember { mutableStateOf<List<DailySummaryEntity>>(emptyList()) }
    var factors by remember { mutableStateOf<List<FactorResult>>(emptyList()) }
    var loaded by remember { mutableStateOf(false) }

    LaunchedEffect(range) {
        SessionSync.sync(context)
        val today = LocalDate.now()
        summaries = db.dailySummaryDao().getRange(today.minusDays(range.days - 1), today).filter { it.isComplete }
        factors = computeFactorAnalysis(db, today)
        loaded = true
    }

    Column(modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(Spacing.lg)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            RangeChip(stringResource(R.string.stats_range_week), range == StatsRange.WEEK) { range = StatsRange.WEEK }
            RangeChip(stringResource(R.string.stats_range_month), range == StatsRange.MONTH) { range = StatsRange.MONTH }
            RangeChip(stringResource(R.string.stats_range_year), range == StatsRange.YEAR) { range = StatsRange.YEAR }
        }
        Spacer(modifier = Modifier.height(Spacing.lg))

        if (!loaded) return@Column

        if (summaries.isEmpty()) {
            Text(text = stringResource(R.string.stats_empty), color = TextSecondary, fontSize = 13.sp)
        } else {
            val avgSleep = summaries.map { it.sleepDurationS }.average()
            val totalGrind = summaries.sumOf { it.mildCount + it.severeCount }
            val totalSnore = summaries.sumOf { it.snoreLightCount + it.snoreLoudCount }

            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xl)) {
                SummaryCell(stringResource(R.string.stats_avg_sleep), formatDurationThai(avgSleep))
                SummaryCell(stringResource(R.string.stats_total_grind), "$totalGrind")
                SummaryCell(stringResource(R.string.stats_total_snore), "$totalSnore")
            }

            Spacer(modifier = Modifier.height(Spacing.xl))
            TrendCanvas(summaries = summaries.sortedBy { it.date }, modifier = Modifier.fillMaxWidth().height(160.dp))
        }

        Spacer(modifier = Modifier.height(Spacing.xxl))
        Text(
            text = stringResource(R.string.stats_factor_title),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(Spacing.xs))
        Text(text = stringResource(R.string.stats_factor_subtitle), color = TextSecondary, fontSize = 12.sp)
        Spacer(modifier = Modifier.height(Spacing.md))

        factors.forEach { factor ->
            FactorRow(factor)
            Spacer(modifier = Modifier.height(Spacing.sm))
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
private fun SummaryCell(label: String, value: String) {
    Column {
        Text(text = value, style = tabularNumberStyle(fontSize = 20.sp), color = Accent)
        Text(text = label, color = TextSecondary, fontSize = 11.sp)
    }
}

@Composable
private fun TrendCanvas(summaries: List<DailySummaryEntity>, modifier: Modifier = Modifier) {
    val maxValue = (summaries.maxOfOrNull { it.mildCount + it.severeCount } ?: 0).coerceAtLeast(1)
    Canvas(modifier = modifier) {
        val barCount = summaries.size
        val gapPx = (1.dp.toPx()).coerceAtMost(size.width / (barCount * 4f).coerceAtLeast(1f))
        val totalGap = gapPx * (barCount - 1).coerceAtLeast(0)
        val barWidth = ((size.width - totalGap) / barCount).coerceAtLeast(0.5f)
        val chartHeight = size.height

        summaries.forEachIndexed { index, day ->
            val count = day.mildCount + day.severeCount
            val fraction = count / maxValue.toFloat()
            val barHeight = (fraction * chartHeight).coerceAtLeast(2f)
            val x = index * (barWidth + gapPx)
            val color = when {
                day.severeCount > 0 -> ErrorRed
                count > 0 -> Warn
                else -> Accent.copy(alpha = 0.25f)
            }
            drawRoundRect(
                color = color,
                topLeft = Offset(x, chartHeight - barHeight),
                size = Size(barWidth, barHeight),
                cornerRadius = CornerRadius(2.dp.toPx(), 2.dp.toPx()),
            )
        }
    }
}

@Composable
private fun FactorRow(factor: FactorResult) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = Spacing.xs),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(text = stringResource(factor.tag.labelRes()), color = MaterialTheme.colorScheme.onBackground, fontSize = 13.sp)
        val insufficient = factor.withCount < MIN_DAYS_PER_GROUP || factor.withoutCount < MIN_DAYS_PER_GROUP
        if (insufficient) {
            Text(
                text = stringResource(R.string.stats_factor_insufficient, minOf(factor.withCount, factor.withoutCount)),
                color = TextSecondary,
                fontSize = 12.sp,
            )
        } else {
            val diffPercent = if (factor.withoutAvg > 0) ((factor.withAvg - factor.withoutAvg) / factor.withoutAvg * 100) else 0.0
            val color = if (diffPercent > 0) ErrorRed else Accent
            Text(
                text = stringResource(
                    R.string.stats_factor_result,
                    "%.1f".format(factor.withAvg),
                    "%.1f".format(factor.withoutAvg),
                ),
                color = color,
                fontSize = 12.sp,
            )
        }
    }
}

private suspend fun computeFactorAnalysis(db: BruxEchoDatabase, today: LocalDate): List<FactorResult> {
    val since = today.minusDays(FACTOR_WINDOW_DAYS - 1)
    val recordedDays = db.dailySummaryDao().getRange(since, today).filter { it.isComplete }.associateBy { it.date }
    val tagsByDate = db.lifestyleTagDao().getSince(since).groupBy({ it.date }, { it.tag })

    return LifestyleTag.entries.map { tag ->
        val withDates = recordedDays.keys.filter { date -> tag in (tagsByDate[date] ?: emptyList()) }
        val withoutDates = recordedDays.keys - withDates.toSet()
        val withAvg = withDates.map { recordedDays.getValue(it).let { s -> s.mildCount + s.severeCount } }
        val withoutAvg = withoutDates.map { recordedDays.getValue(it).let { s -> s.mildCount + s.severeCount } }
        FactorResult(
            tag = tag,
            withAvg = if (withAvg.isNotEmpty()) withAvg.average() else 0.0,
            withoutAvg = if (withoutAvg.isNotEmpty()) withoutAvg.average() else 0.0,
            withCount = withDates.size,
            withoutCount = withoutDates.size,
        )
    }
}
