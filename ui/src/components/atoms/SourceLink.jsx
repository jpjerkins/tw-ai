import { Link } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

export default function SourceLink({ url, title }) {
  // Add https:// only if the URL doesn't already have a protocol
  const full_url = url.startsWith('http://') || url.startsWith('https://')
    ? url
    : `https://${url}`;
  return (
    <Link
      href={full_url}
      target="_blank"
      rel="noopener noreferrer"
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.5,
        textDecoration: 'none',
        '&:hover': {
          textDecoration: 'underline',
        },
      }}
    >
      {title}
      <OpenInNewIcon sx={{ fontSize: 16 }} />
    </Link>
  );
}
