import { Box } from '@mui/material';
import QueryTextField from '../atoms/QueryTextField';
import SubmitButton from '../atoms/SubmitButton';

export default function QueryForm({ query, onQueryChange, onSubmit, loading }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: 'flex',
        gap: 2,
        alignItems: 'flex-start',
      }}
    >
      <QueryTextField
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        disabled={loading}
        onKeyPress={handleKeyPress}
        placeholder="e.g., How do I create a custom widget?"
      />
      <SubmitButton
        onClick={handleSubmit}
        disabled={!query.trim()}
        loading={loading}
      />
    </Box>
  );
}
