export function PlateIcon({ size = 56 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" stroke="var(--line)" strokeWidth="1.5" aria-hidden>
      <circle cx="32" cy="32" r="26" />
      <circle cx="32" cy="32" r="17" />
    </svg>
  );
}
