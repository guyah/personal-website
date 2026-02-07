const STORAGE_KEY = "talk-state-v1";

const DEFAULT_STATE = {
	isOpen: false,
	connection: "disconnected",
	status: "Press Start, speak, then Stop.",
	statusLive: false,
	micState: "idle",
	botState: "standing by",
	speaking: false,
	transcript: "",
	reply: "",
	permission: "",
	latencies: {
		stt: "—",
		llm: "—",
		ttsFirst: "—",
		ttsTotal: "—",
	},
	startDisabled: false,
	stopDisabled: true,
};

let state = { ...DEFAULT_STATE };
let hydrated = false;
const listeners = new Set();
let saveTimer = null;

function canUseStorage() {
	return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function scheduleSave() {
	if (!canUseStorage()) return;
	if (saveTimer) return;
	saveTimer = window.setTimeout(() => {
		saveTimer = null;
		try {
			window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
		} catch {
			// ignore storage failures
		}
	}, 200);
}

function load() {
	if (!canUseStorage()) return;
	try {
		const raw = window.localStorage.getItem(STORAGE_KEY);
		if (!raw) return;
		const parsed = JSON.parse(raw);
		state = {
			...DEFAULT_STATE,
			...parsed,
			latencies: {
				...DEFAULT_STATE.latencies,
				...(parsed.latencies || {}),
			},
		};
	} catch {
		// ignore malformed storage
	}
}

function hydrate() {
	if (hydrated) return;
	hydrated = true;
	load();
}

function notify() {
	listeners.forEach((fn) => fn(state));
	scheduleSave();
}

function setState(partial) {
	state = {
		...state,
		...partial,
		latencies: {
			...state.latencies,
			...(partial.latencies || {}),
		},
	};
	notify();
}

function getState() {
	return state;
}

function subscribe(fn) {
	listeners.add(fn);
	fn(state);
	return () => listeners.delete(fn);
}

export const talkStore = {
	hydrate,
	getState,
	setState,
	subscribe,
	DEFAULT_STATE,
};
