import { Box, Typography, Paper, Alert } from '@mui/material';

export default function ResponseDisplay({ response, error }) {
  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 3 }}>
        {error}
      </Alert>
    );
  }

  if (!response) {
    return null;
  }

  return (
    <Paper
      elevation={2}
      sx={{
        p: 3,
        mt: 3,
        backgroundColor: 'background.paper',
      }}
    >
      <Typography
        variant="body1"
        sx={{
          whiteSpace: 'pre-wrap',
          lineHeight: 1.6,
        }}
      >
        {response}
      </Typography>
    </Paper>
  );
}
