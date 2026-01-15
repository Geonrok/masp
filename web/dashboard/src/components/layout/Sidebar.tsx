import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  ChartBarIcon,
  ClockIcon,
  CogIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';

const navItems = [
  { to: '/', icon: HomeIcon, label: 'Dashboard' },
  { to: '/positions', icon: ChartBarIcon, label: 'Positions' },
  { to: '/trades', icon: ClockIcon, label: 'Trades' },
  { to: '/strategies', icon: BoltIcon, label: 'Strategies' },
  { to: '/settings', icon: CogIcon, label: 'Settings' },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-900 text-white">
      <div className="p-4 text-xl font-bold border-b border-gray-700">
        MASP Dashboard
      </div>
      <nav className="mt-4">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center px-4 py-3 hover:bg-gray-800 ${
                isActive ? 'bg-gray-800 border-l-4 border-blue-500' : ''
              }`
            }
          >
            <Icon className="w-5 h-5 mr-3" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
