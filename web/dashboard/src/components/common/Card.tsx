import type { ReactNode } from 'react';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export default function Card({ title, children, className = '' }: CardProps) {
  return (
    <div className={`bg-white rounded shadow p-4 ${className}`}>
      {title ? <div className="text-sm text-gray-500 mb-2">{title}</div> : null}
      {children}
    </div>
  );
}
