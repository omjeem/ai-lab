import { describe, it, expect } from 'vitest';
import { parseDevice } from '@/lib/deviceParsing';

const DESKTOP_CHROME =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) ' +
  'Chrome/120.0.0.0 Safari/537.36';

const MOBILE_SAFARI =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) ' +
  'Version/17.0 Mobile/15E148 Safari/604.1';

describe('parseDevice', () => {
  it('returns null with no user-agent header at all', () => {
    expect(parseDevice(null)).toBeNull();
  });

  it('parses a desktop browser and defaults deviceType to desktop', () => {
    const device = parseDevice(DESKTOP_CHROME);
    expect(device).not.toBeNull();
    expect(device?.browser).toBe('Chrome');
    expect(device?.os).toBe('Mac OS');
    expect(device?.deviceType).toBe('desktop');
    expect(device?.isBot).toBe(false);
  });

  it('parses a mobile browser with its real device type, not the desktop default', () => {
    const device = parseDevice(MOBILE_SAFARI);
    expect(device?.browser).toBe('Mobile Safari');
    expect(device?.os).toBe('iOS');
    expect(device?.deviceType).toBe('mobile');
  });

  it('flags known crawler user-agents as bots', () => {
    const device = parseDevice('Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)');
    expect(device?.isBot).toBe(true);
  });
});
