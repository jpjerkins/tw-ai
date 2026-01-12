import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import QueryInterface from './components/organisms/QueryInterface';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryInterface />
    </ThemeProvider>
  );
}

export default App
