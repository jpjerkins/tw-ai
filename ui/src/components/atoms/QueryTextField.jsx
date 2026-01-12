import { TextField } from '@mui/material';

export default function QueryTextField({ value, onChange, disabled, ...props }) {
  return (
    <TextField
      fullWidth
      variant="outlined"
      label="Enter your question"
      value={value}
      onChange={onChange}
      disabled={disabled}
      {...props}
    />
  );
}
