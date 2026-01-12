import { Button } from '@mui/material';

export default function SubmitButton({ onClick, disabled, loading, children = 'Submit' }) {
  return (
    <Button
      variant="contained"
      color="primary"
      onClick={onClick}
      disabled={disabled || loading}
      sx={{ minWidth: '120px' }}
    >
      {loading ? 'Loading...' : children}
    </Button>
  );
}
