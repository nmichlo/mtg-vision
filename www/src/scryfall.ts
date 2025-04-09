import {Detection, ScryfallCardSymbol, ScryfallSymbolsResponse} from "./types";
import {html, TemplateResult} from "lit";

// map e.g. "{E}" to symbol data
const symbolCache = new Map<string, ScryfallCardSymbol>();


export async function fetchSymbology() {
  if (symbolCache.size !== 0) {
    return
  }
  // Only fetch if the cache is empty
  try {
    const response = await fetch('https://api.scryfall.com/symbology');
    const data: ScryfallSymbolsResponse = await response.json();
    data.data.forEach(symbol => {symbolCache.set(symbol.symbol, symbol);});
    console.log('Loaded', symbolCache.size, 'symbols from Scryfall API');
  } catch (error) {
    console.error('Error fetching symbology:', error);
  }
}

export function replaceSymbols(text: string): (string | TemplateResult)[] {
  if (!text) {
    return [];
  }
  if (!symbolCache.size) {
    console.warn('Symbol cache is empty. Please fetch scryfall symbology first.');
    return [text];
  }
  // 1. for each character
  // if "{", continue until "}"
  // 2. check if the symbol exists in the cache
  // 3. if it exists, replace with the SVG URL
  // 4. if it doesn't exist, replace with the original text
  const items = [];
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '{') {
      let symbol = '{';
      while (text[++i] !== '}') {
        symbol += text[i];
      }
      symbol += '}';
      const symbolData = symbolCache.get(symbol);
      if (symbolData) {
        items.push(html`<img src="${symbolData.svg_uri}" alt="${symbol}" style="width: 16px; height: 16px; vertical-align: middle;">`);
      } else {
        items.push(`{${symbol}}`);
      }
    } else {
      items.push(char);
    }
  }
  return items
}


export function augmentDetections(detections: Detection[]): Detection[] {
  return detections.map(det => {
    if (det.matches) {
      det.matches = det.matches.map(match => {
        if (!match.extra_data) {
          match.extra_data = {
            oracle_text: replaceSymbols(match?.all_data?.oracle_text || ''),
            mana_cost: replaceSymbols(match?.all_data?.mana_cost || ''),
          };
        }
        return match;
      })
    }
    return det;
  });
}
