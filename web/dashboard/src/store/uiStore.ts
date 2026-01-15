import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { useShallow } from 'zustand/react/shallow';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: number;
}

interface UIState {
  sidebarCollapsed: boolean;
  activeModal: string | null;
  notifications: Notification[];
}

interface UIActions {
  toggleSidebar: () => void;
  openModal: (modalId: string) => void;
  closeModal: () => void;
  addNotification: (type: Notification['type'], message: string) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

const generateId = (): string => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

const getInitialState = (): UIState => ({
  sidebarCollapsed: false,
  activeModal: null,
  notifications: [],
});

export const useUIStore = create<UIState & UIActions>()(
  immer((set) => ({
    ...getInitialState(),

    toggleSidebar: () =>
      set((state) => {
        state.sidebarCollapsed = !state.sidebarCollapsed;
      }),

    openModal: (modalId) =>
      set((state) => {
        state.activeModal = modalId;
      }),

    closeModal: () =>
      set((state) => {
        state.activeModal = null;
      }),

    addNotification: (type, message) =>
      set((state) => {
        state.notifications.push({
          id: generateId(),
          type,
          message,
          timestamp: Date.now(),
        });
      }),

    removeNotification: (id) =>
      set((state) => {
        state.notifications = state.notifications.filter((n) => n.id !== id);
      }),

    clearNotifications: () =>
      set((state) => {
        state.notifications = [];
      }),
  }))
);

export const useSidebarCollapsed = () => useUIStore((state) => state.sidebarCollapsed);
export const useActiveModal = () => useUIStore((state) => state.activeModal);
export const useNotifications = () => useUIStore(useShallow((state) => state.notifications));
