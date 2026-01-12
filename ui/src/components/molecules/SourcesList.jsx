import { Box, Typography, Paper } from '@mui/material';
import SourceLink from '../atoms/SourceLink';

export default function SourcesList({ sources }) {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        backgroundColor: 'grey.50',
      }}
    >
      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
        Sources:
      </Typography>
      <Box component="ul" sx={{ m: 0, pl: 2 }}>
        {sources.map((source, index) => (
          <Box component="li" key={index} sx={{ mb: 0.5 }}>
            <SourceLink url={source.link_url} title={source.title} />
            {source.rank && (
              <Typography
                component="span"
                variant="caption"
                color="text.secondary"
                sx={{ ml: 1 }}
              >
                (relevance: {source.rank.toFixed(3)})
              </Typography>
            )}
          </Box>
        ))}
      </Box>
    </Paper>
  );
}
