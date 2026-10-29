package com.bruxecho.recorder.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
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
import com.bruxecho.recorder.playback.rememberClipPlayer
import com.bruxecho.recorder.ui.theme.Accent
import com.bruxecho.recorder.ui.theme.Spacing
import com.bruxecho.recorder.ui.theme.Surface1
import com.bruxecho.recorder.ui.theme.Surface2
import com.bruxecho.recorder.ui.theme.TextSecondary
import com.bruxecho.recorder.ui.theme.Warn
import com.bruxecho.recorder.ui.theme.tabularNumberStyle

private const val ASSET_DIR = "reference_clips"

/**
 * Known recordings of real sleep bruxism, for the user to learn what the
 * sound is — **not** detections.
 *
 * Kept as its own screen rather than a section of the results screen, and
 * loaded from bundled assets rather than from any session, so there is no
 * path by which one of these can be counted as an event. The banner says so
 * in words too: the model currently finds roughly two thirds of known
 * grinding, so a user comparing "what it sounds like" against "what my night
 * produced" has to be able to tell which list is which.
 */
@Composable
fun ReferenceClipsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val player = rememberClipPlayer()
    val clips = remember {
        (context.assets.list(ASSET_DIR) ?: emptyArray()).filter { it.endsWith(".wav") }.sorted()
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(Spacing.sm),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = stringResource(R.string.miccheck_back),
                    tint = MaterialTheme.colorScheme.onBackground,
                )
            }
            Text(
                text = stringResource(R.string.reference_clips_title),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
        }

        // Deliberately loud and above the list: this screen must never be
        // mistaken for last night's results.
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = Spacing.lg)
                .clip(RoundedCornerShape(12.dp))
                .background(Warn.copy(alpha = 0.15f))
                .padding(Spacing.md),
        ) {
            Text(
                text = stringResource(R.string.reference_clips_not_results),
                color = Warn,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(Spacing.xs))
            Text(
                text = stringResource(R.string.reference_clips_banner_body),
                color = TextSecondary,
                fontSize = 12.sp,
            )
        }

        Spacer(modifier = Modifier.height(Spacing.md))
        Text(
            text = stringResource(R.string.reference_clips_intro),
            color = TextSecondary,
            fontSize = 13.sp,
            modifier = Modifier.padding(horizontal = Spacing.lg),
        )
        Spacer(modifier = Modifier.height(Spacing.md))

        LazyColumn(
            modifier = Modifier.fillMaxWidth().padding(horizontal = Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            items(clips) { name ->
                val path = "asset:///$ASSET_DIR/$name"
                val isPlaying = player.currentlyPlayingPath == path
                ReferenceClipRow(
                    index = clips.indexOf(name) + 1,
                    isPlaying = isPlaying,
                    onToggle = { if (isPlaying) player.stop() else player.play(path) },
                )
            }
            item { Spacer(modifier = Modifier.height(Spacing.xl)) }
        }
    }
}

@Composable
private fun ReferenceClipRow(index: Int, isPlaying: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (isPlaying) Surface2 else Surface1)
            .clickable(onClick = onToggle)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.reference_clip_label, index),
                style = tabularNumberStyle(fontSize = 15.sp),
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                text = stringResource(R.string.reference_clip_source),
                color = TextSecondary,
                fontSize = 12.sp,
            )
        }
        Icon(
            imageVector = if (isPlaying) Icons.Filled.Stop else Icons.Filled.PlayArrow,
            contentDescription = null,
            tint = Accent,
            modifier = Modifier.size(28.dp),
        )
    }
}
