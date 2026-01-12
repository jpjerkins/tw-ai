import { Box, Paper, Alert } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
      <Box
        sx={{
          '& h1': {
            fontSize: '2rem',
            fontWeight: 600,
            mb: 2,
            mt: 3,
            '&:first-of-type': { mt: 0 },
          },
          '& h2': {
            fontSize: '1.5rem',
            fontWeight: 600,
            mb: 1.5,
            mt: 2.5,
          },
          '& h3': {
            fontSize: '1.25rem',
            fontWeight: 600,
            mb: 1,
            mt: 2,
          },
          '& p': {
            mb: 1.5,
            lineHeight: 1.7,
          },
          '& ul, & ol': {
            mb: 1.5,
            pl: 3,
          },
          '& li': {
            mb: 0.5,
          },
          '& code': {
            backgroundColor: 'grey.100',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.875em',
            fontFamily: 'monospace',
          },
          '& pre': {
            backgroundColor: 'grey.100',
            p: 2,
            borderRadius: '4px',
            overflow: 'auto',
            mb: 2,
          },
          '& pre code': {
            backgroundColor: 'transparent',
            padding: 0,
          },
          '& blockquote': {
            borderLeft: '4px solid',
            borderColor: 'primary.main',
            pl: 2,
            ml: 0,
            fontStyle: 'italic',
            color: 'text.secondary',
          },
          '& a': {
            color: 'primary.main',
            textDecoration: 'none',
            '&:hover': {
              textDecoration: 'underline',
            },
          },
          '& table': {
            width: '100%',
            borderCollapse: 'collapse',
            mb: 2,
          },
          '& th, & td': {
            border: '1px solid',
            borderColor: 'divider',
            p: 1,
            textAlign: 'left',
          },
          '& th': {
            backgroundColor: 'grey.100',
            fontWeight: 600,
          },
        }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {response}
        </ReactMarkdown>
      </Box>
    </Paper>
  );
}
