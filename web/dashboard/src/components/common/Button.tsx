import type { ButtonHTMLAttributes } from 'react';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement>;

export default function Button({ className = '', ...props }: ButtonProps) {
  return (
    <button
      type="button"
      className={`px-4 py-2 rounded font-medium ${className}`}
      {...props}
    />
  );
}
