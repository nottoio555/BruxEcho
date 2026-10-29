package com.bruxecho.recorder.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.bruxecho.recorder.R
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.BgDeep
import com.bruxecho.recorder.ui.theme.TextSecondary

/** The "บันทึก" bottom-nav tab: ปฏิทิน (calendar) + สถิติ (stats) sub-tabs. */
@Composable
fun LogScreen(onOpenSessionDetail: (String) -> Unit, onOpenResults: (String) -> Unit) {
    var subTab by remember { mutableIntStateOf(0) }

    Column(modifier = Modifier.fillMaxSize()) {
        // No custom `indicator` override — Material3's default TabRow indicator
        // already reads its color from `contentColor` and positions itself
        // correctly; a hand-rolled one needs `Modifier.tabIndicatorOffset`,
        // which isn't resolvable against this Compose BOM.
        TabRow(
            selectedTabIndex = subTab,
            containerColor = BgDeep,
            contentColor = Accent,
        ) {
            Tab(
                selected = subTab == 0,
                onClick = { subTab = 0 },
                text = { Text(text = stringResource(R.string.log_tab_calendar)) },
                selectedContentColor = Accent,
                unselectedContentColor = TextSecondary,
            )
            Tab(
                selected = subTab == 1,
                onClick = { subTab = 1 },
                text = { Text(text = stringResource(R.string.log_tab_stats)) },
                selectedContentColor = Accent,
                unselectedContentColor = TextSecondary,
            )
        }

        when (subTab) {
            0 -> CalendarScreen(onOpenSessionDetail = onOpenSessionDetail, onOpenResults = onOpenResults)
            else -> StatsScreen()
        }
    }
}
