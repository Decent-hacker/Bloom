const paths = {
  camera: <><path d="M4 7h3l1.4-2h7.2L17 7h3a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"/><circle cx="12" cy="13" r="3.5"/></>,
  upload: <><path d="M12 16V4m0 0-4 4m4-4 4 4"/><path d="M5 14v5h14v-5"/></>,
  shield: <><path d="M12 3 4.5 6v5.5c0 4.6 3.1 7.8 7.5 9.5 4.4-1.7 7.5-4.9 7.5-9.5V6L12 3Z"/><path d="m9 12 2 2 4-4"/></>,
  trash: <><path d="M4 7h16M10 11v5m4-5v5M6 7l1 14h10l1-14M9 7V4h6v3"/></>,
  logout: <><path d="M10 5H5v14h5"/><path d="m14 16 4-4-4-4m4 4H9"/></>,
  arrow: <path d="m9 18 6-6-6-6"/>,
};

function Icon({ name, size = 20, className = "" }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className}>{paths[name]}</svg>;
}

export default Icon;
