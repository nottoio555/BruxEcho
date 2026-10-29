package com.bruxecho.recorder.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bruxecho.recorder.R
import com.bruxecho.recorder.data.db.LifestyleTag
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.AccentDim
import com.bruxecho.recorder.ui.theme.OnAccent
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.TextSecondary

private val TagIconSize = 32.dp
private val TagIconCircle = 60.dp

/**
 * Asked once before recording starts: what happened today that might explain
 * tonight. Never blocks recording — [onDone] always fires, with an empty set
 * if the user skips, and the caller starts recording either way.
 *
 * A full page rather than the half-height bottom sheet it began as. The sheet
 * had to fit five options into a horizontally scrolling chip row, so two of
 * them sat off-screen and the whole thing read as a dismissible interruption.
 * At full height the options are a grid that fits at once, each with a drawn
 * icon large enough to recognize without reading.
 *
 * Tags are saved against *today's date*, not a session id, because the
 * recording has not started yet and no session id exists — see the
 * `lifestyle_tags` table, keyed by date.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LifestyleTagScreen(
    onDone: (selectedTags: Set<LifestyleTag>, memo: String) -> Unit,
    onCancel: () -> Unit,
) {
    var selected by remember { mutableStateOf(setOf<LifestyleTag>()) }
    var memo by remember { mutableStateOf("") }

    // Both buttons at the bottom start recording, so without this there is no
    // way out of the page — the bottom sheet it replaced could always just be
    // swiped away, and system Back would otherwise leave the app entirely.
    BackHandler(onBack = onCancel)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .systemBarsPadding()
            .padding(horizontal = Spacing.lg),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = stringResource(R.string.lifestyle_sheet_title),
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.weight(1f),
            )
            IconButton(onClick = onCancel) {
                Icon(
                    imageVector = Icons.Filled.Close,
                    contentDescription = stringResource(R.string.lifestyle_cancel),
                    tint = TextSecondary,
                )
            }
        }
        Spacer(modifier = Modifier.height(Spacing.xs))
        Text(
            text = stringResource(R.string.lifestyle_sheet_subtitle),
            color = TextSecondary,
            fontSize = 14.sp,
        )

        Spacer(modifier = Modifier.height(Spacing.lg))
        LazyVerticalGrid(
            columns = GridCells.Fixed(3),
            modifier = Modifier.fillMaxWidth().weight(1f),
            horizontalArrangement = Arrangement.spacedBy(Spacing.md),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            items(LifestyleTag.entries) { tag ->
                TagTile(
                    tag = tag,
                    isSelected = tag in selected,
                    onToggle = {
                        selected = if (tag in selected) selected - tag else selected + tag
                    },
                )
            }
        }

        Spacer(modifier = Modifier.height(Spacing.md))
        OutlinedTextField(
            value = memo,
            onValueChange = { memo = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text(text = stringResource(R.string.lifestyle_sheet_memo_hint)) },
            colors = TextFieldDefaults.colors(
                focusedIndicatorColor = Accent,
                cursorColor = Accent,
            ),
            singleLine = true,
        )

        Spacer(modifier = Modifier.height(Spacing.lg))
        Button(
            onClick = { onDone(selected, memo.trim()) },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Accent, contentColor = OnAccent),
        ) {
            Text(text = stringResource(R.string.lifestyle_sheet_confirm), fontSize = 16.sp)
        }
        Spacer(modifier = Modifier.height(Spacing.xs))
        TextButton(
            onClick = { onDone(emptySet(), "") },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(text = stringResource(R.string.lifestyle_sheet_skip), color = TextSecondary)
        }
        Spacer(modifier = Modifier.height(Spacing.md))
    }
}

@Composable
private fun TagTile(tag: LifestyleTag, isSelected: Boolean, onToggle: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(16.dp))
            .background(if (isSelected) AccentDim else Surface1)
            .border(
                width = if (isSelected) 2.dp else 0.dp,
                color = if (isSelected) Accent else Surface1,
                shape = RoundedCornerShape(16.dp),
            )
            .clickable(onClick = onToggle)
            .padding(Spacing.sm),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Box(contentAlignment = Alignment.Center) {
            Box(
                modifier = Modifier
                    .size(TagIconCircle)
                    .clip(CircleShape)
                    .background(if (isSelected) Accent.copy(alpha = 0.22f) else MaterialTheme.colorScheme.background),
            )
            Icon(
                imageVector = tag.icon(),
                contentDescription = null,
                tint = if (isSelected) Accent else TextSecondary,
                modifier = Modifier.size(TagIconSize),
            )
            if (isSelected) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(22.dp)
                        .clip(CircleShape)
                        .background(Accent),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Filled.Check,
                        contentDescription = null,
                        tint = OnAccent,
                        modifier = Modifier.size(14.dp),
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(Spacing.sm))
        Text(
            text = stringResource(tag.labelRes()),
            color = if (isSelected) MaterialTheme.colorScheme.onBackground else TextSecondary,
            fontSize = 13.sp,
            fontWeight = if (isSelected) FontWeight.Medium else FontWeight.Normal,
            textAlign = TextAlign.Center,
        )
    }
}
