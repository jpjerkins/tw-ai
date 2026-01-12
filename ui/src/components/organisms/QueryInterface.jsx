import { useState } from 'react';
import { Container, Box, Typography, CircularProgress } from '@mui/material';
import QueryForm from '../molecules/QueryForm';
import ResponseDisplay from '../molecules/ResponseDisplay';
import SourcesList from '../molecules/SourcesList';
import { askQuestion } from '../../services/api';

export default function QueryInterface() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [sources, setSources] = useState([]);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    setSources([]);

    try {
      const result = await askQuestion(query);
      setResponse(result.answer);
      setSources(result.sources);
    } catch (err) {
      setError(err.message || 'An error occurred while processing your question');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          TiddlyWiki AI Assistant
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Ask questions about TiddlyWiki and get answers from indexed documentation
        </Typography>
      </Box>

      <QueryForm
        query={query}
        onQueryChange={setQuery}
        onSubmit={handleSubmit}
        loading={loading}
      />

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      )}

      <ResponseDisplay response={response} error={error} />

      {sources.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <SourcesList sources={sources} />
        </Box>
      )}
    </Container>
  );
}
