package com.bruxecho.recorder.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.data.SessionSync
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.DailySummaryEntity
import com.bruxecho.recorder.data.db.DaySession
import com.bruxecho.recorder.data.db.EventType
import com.bruxecho.recorder.data.db.LifestyleTagEntity
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.ErrorRed
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.Positive
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.Surface2
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.ui.theme.tabularNumberStyle
import com.bruxecho.recorder.util.formatDurationThai
import com.bruxecho.recorder.util.formatThaiMonthYear
import java.time.LocalDate
import java.time.YearMonth
import java.time.ZoneId
import java.time.format.TextStyle
import java.util.Locale

private val THAI_LOCALE: Locale = Locale.Builder().setLanguage("th").setRegion("TH").build()
private val THAI_WEEKDAY_LABELS = listOf("อา", "จ", "อ", "พ", "พฤ", "ศ", "ส")

private fun dotColorFor(summary: DailySummaryEntity): Color = when {
    !summary.isComplete -> TextSecondary
    summary.severeCount > 0 -> ErrorRed
    summary.mildCount > 0 -> Warn
    else -> Positive
}

@Composable
fun CalendarScreen(onOpenSessionDetail: (String) -> Unit, onOpenResults: (String) -> Unit) {
    val context = LocalContext.current
    val db = remember { BruxEchoDatabase.getInstance(context) }
    var visibleMonth by remember { mutableStateOf(YearMonth.now()) }
    var summaries by remember { mutableStateOf<Map<LocalDate, DailySummaryEntity>>(emptyMap()) }
    var selectedDate by remember { mutableStateOf<LocalDate?>(null) }
    var selectedTags by remember { mutableStateOf<List<LifestyleTagEntity>>(emptyList()) }
    var selectedSessions by remember { mutableStateOf<List<DaySession>>(emptyList()) }

    LaunchedEffect(visibleMonth) {
        SessionSync.sync(context)
        val start = visibleMonth.atDay(1)
        val end = visibleMonth.atEndOfMonth()
        summaries = db.dailySummaryDao().getRange(start, end).associateBy { it.date }
    }

    LaunchedEffect(selectedDate) {
        val date = selectedDate
        selectedTags = if (date != null) db.lifestyleTagDao().getForDate(date) else emptyList()
        selectedSessions = if (date == null) {
            emptyList()
        } else {
            // A "day" is a timezone question, and `sessions` stores instants —
            // so the range is built here, where the zone is known.
            val zone = ZoneId.systemDefault()
            val startMs = date.atStartOfDay(zone).toInstant().toEpochMilli()
            val endMs = date.plusDays(1).atStartOfDay(zone).toInstant().toEpochMilli()
            db.sessionDao().getStartedBetween(startMs, endMs).map { session ->
                DaySession(session, db.eventDao().countForSession(session.sessionId, EventType.GRIND))
            }
        }
    }

    // Scrollable because the detail card grows with the number of sessions
    // that day: with three, the list ran off the bottom of the screen and the
    // rows below the first were unreachable. The grid below has a fixed
    // height, so nesting it here doesn't hit the infinite-constraint problem.
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(Spacing.lg)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { visibleMonth = visibleMonth.minusMonths(1); selectedDate = null }) {
                Icon(imageVector = Icons.Filled.ChevronLeft, contentDescription = null, tint = TextSecondary)
            }
            Text(
                text = formatThaiMonthYear(visibleMonth),
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            IconButton(onClick = { visibleMonth = visibleMonth.plusMonths(1); selectedDate = null }) {
                Icon(imageVector = Icons.Filled.ChevronRight, contentDescription = null, tint = TextSecondary)
            }
        }

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
            THAI_WEEKDAY_LABELS.forEach { label ->
                Text(text = label, color = TextSecondary, fontSize = 12.sp, modifier = Modifier.size(36.dp), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            }
        }

        val firstDayOffset = (visibleMonth.atDay(1).dayOfWeek.value % 7) // Mon=1..Sun=7 -> Sun=0..Sat=6
        val daysInMonth = visibleMonth.lengthOfMonth()
        val cells: List<LocalDate?> = List(firstDayOffset) { null } + (1..daysInMonth).map { visibleMonth.atDay(it) }

        LazyVerticalGrid(
            columns = GridCells.Fixed(7),
            modifier = Modifier.fillMaxWidth().height(((cells.size / 7 + 1) * 48).dp),
        ) {
            items(cells) { date ->
                if (date == null) {
                    Box(modifier = Modifier.aspectRatio(1f))
                } else {
                    DayCell(
                        date = date,
                        summary = summaries[date],
                        isSelected = date == selectedDate,
                        onClick = { selectedDate = if (selectedDate == date) null else date },
                    )
                }
            }
        }

        selectedDate?.let { date ->
            Spacer(modifier = Modifier.height(Spacing.md))
            DateDetailCard(
                date = date,
                summary = summaries[date],
                tags = selectedTags,
                sessions = selectedSessions,
                onOpenSessionDetail = onOpenSessionDetail,
                onOpenResults = onOpenResults,
            )
        }
    }
}

@Composable
private fun DayCell(date: LocalDate, summary: DailySummaryEntity?, isSelected: Boolean, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .aspectRatio(1f)
            .padding(2.dp)
            .clip(CircleShape)
            .background(if (isSelected) Accent.copy(alpha = 0.25f) else Color.Transparent)
            .clickable(onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = date.dayOfMonth.toString(),
            color = if (date == LocalDate.now()) Accent else MaterialTheme.colorScheme.onBackground,
            fontSize = 13.sp,
            fontWeight = if (date == LocalDate.now()) FontWeight.Bold else FontWeight.Normal,
        )
        Spacer(modifier = Modifier.height(2.dp))
        if (summary != null) {
            Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(dotColorFor(summary)))
        } else {
            Spacer(modifier = Modifier.size(6.dp))
        }
    }
}

@Composable
private fun DateDetailCard(
    date: LocalDate,
    summary: DailySummaryEntity?,
    tags: List<LifestyleTagEntity>,
    sessions: List<DaySession>,
    onOpenSessionDetail: (String) -> Unit,
    onOpenResults: (String) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Surface1)
            .padding(Spacing.lg),
    ) {
        Text(
            text = date.dayOfWeek.getDisplayName(TextStyle.FULL, THAI_LOCALE).replaceFirstChar { it.uppercase() } + " " + date.dayOfMonth,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(modifier = Modifier.height(Spacing.sm))

        if (summary == null) {
            Text(text = stringResource(R.string.calendar_no_data), color = TextSecondary, fontSize = 13.sp)
            return@Column
        }

        if (!summary.isComplete) {
            Text(text = stringResource(R.string.calendar_incomplete), color = TextSecondary, fontSize = 13.sp)
        } else {
            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xl)) {
                Column {
                    Text(text = formatDurationThai(summary.sleepDurationS), style = tabularNumberStyle(fontSize = 18.sp), color = MaterialTheme.colorScheme.onBackground)
                    Text(text = stringResource(R.string.calendar_sleep_duration), color = TextSecondary, fontSize = 11.sp)
                }
                Column {
                    Text(text = "${summary.mildCount}", style = tabularNumberStyle(fontSize = 18.sp), color = Warn)
                    Text(text = stringResource(R.string.calendar_mild_count), color = TextSecondary, fontSize = 11.sp)
                }
                Column {
                    Text(text = "${summary.severeCount}", style = tabularNumberStyle(fontSize = 18.sp), color = ErrorRed)
                    Text(text = stringResource(R.string.calendar_severe_count), color = TextSecondary, fontSize = 11.sp)
                }
            }
        }

        if (tags.isNotEmpty()) {
            Spacer(modifier = Modifier.height(Spacing.md))
            Text(text = stringResource(R.string.calendar_tags_title), color = TextSecondary, fontSize = 12.sp)
            Spacer(modifier = Modifier.height(Spacing.xs))
            val tagLabels = tags.map { entry -> "${entry.tag.emoji()} ${stringResource(entry.tag.labelRes())}" }
            Text(
                text = tagLabels.joinToString(" · "),
                color = MaterialTheme.colorScheme.onBackground,
                fontSize = 13.sp,
            )
        }

        Spacer(modifier = Modifier.height(Spacing.md))

        // Always a list, even for one session: a day can hold several, and the
        // old single-button layout could only ever reach one of them.
        Text(
            text = stringResource(R.string.calendar_sessions_title, sessions.size),
            color = TextSecondary,
            fontSize = 12.sp,
        )
        Spacer(modifier = Modifier.height(Spacing.xs))
        sessions.forEach { entry ->
            SessionRow(
                entry = entry,
                onOpenResults = onOpenResults,
                onOpenSessionDetail = onOpenSessionDetail,
            )
            Spacer(modifier = Modifier.height(Spacing.xs))
        }
    }
}

@Composable
private fun SessionRow(
    entry: DaySession,
    onOpenResults: (String) -> Unit,
    onOpenSessionDetail: (String) -> Unit,
) {
    val zone = remember { ZoneId.systemDefault() }
    val startedAt = entry.session.startedAt.atZone(zone).toLocalTime()
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Surface2)
            .clickable(enabled = entry.session.isComplete) { onOpenResults(entry.session.sessionId) }
            .padding(horizontal = Spacing.md, vertical = Spacing.sm),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "%02d:%02d".format(startedAt.hour, startedAt.minute),
                style = tabularNumberStyle(fontSize = 15.sp),
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                text = if (entry.session.isComplete) {
                    stringResource(R.string.calendar_session_grind_count, entry.grindCount)
                } else {
                    stringResource(R.string.calendar_incomplete)
                },
                color = TextSecondary,
                fontSize = 12.sp,
            )
        }
        OutlinedButton(onClick = { onOpenSessionDetail(entry.session.sessionId) }) {
            Text(text = stringResource(R.string.calendar_view_session), fontSize = 12.sp)
        }
    }
}
