import { TemplateResult } from "lit";

export interface Device {
  deviceId: string;
  label: string;
  kind: string;
}

export interface ScryfallCardData {
  /**
   * EXAMPLE:
   *
   * "object": "card",
   * "id": "b8c69031-588d-44a2-89b3-972a335541e5",
   * "oracle_id": "f96fcb37-5f17-4231-b1a2-64f5fe45730b",
   * "multiverse_ids": [
   *   580879
   * ],
   * "mtgo_id": 108586,
   * "tcgplayer_id": 286354,
   * "cardmarket_id": 675359,
   * "name": "Skorpekh Destroyer",
   * "lang": "en",
   * "released_at": "2022-10-07",
   * "uri": "https://api.scryfall.com/cards/b8c69031-588d-44a2-89b3-972a335541e5",
   * "scryfall_uri": "https://scryfall.com/card/40k/57/skorpekh-destroyer?utm_source=api",
   * "layout": "normal",
   * "highres_image": true,
   * "image_status": "highres_scan",
   * "image_uris": {
   *   "small": "https://cards.scryfall.io/small/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.jpg?1673308827",
   *   "normal": "https://cards.scryfall.io/normal/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.jpg?1673308827",
   *   "large": "https://cards.scryfall.io/large/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.jpg?1673308827",
   *   "png": "https://cards.scryfall.io/png/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.png?1673308827",
   *   "art_crop": "https://cards.scryfall.io/art_crop/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.jpg?1673308827",
   *   "border_crop": "https://cards.scryfall.io/border_crop/front/b/8/b8c69031-588d-44a2-89b3-972a335541e5.jpg?1673308827"
   * },
   * "mana_cost": "{2}{B}{B}",
   * "cmc": 4,
   * "type_line": "Artifact Creature — Necron",
   * "oracle_text": "Deathtouch\nHyperphase Threshers — Whenever an artifact you control enters, this creature gains first strike until end of turn.",
   * "power": "4",
   * "toughness": "2",
   * "colors": [
   *   "B"
   * ],
   * "color_identity": [
   *   "B"
   * ],
   * "keywords": [
   *   "Hyperphase Threshers",
   *   "Deathtouch"
   * ],
   * "legalities": {
   *   "standard": "not_legal",
   *   "future": "not_legal",
   *   "historic": "not_legal",
   *   "timeless": "not_legal",
   *   "gladiator": "not_legal",
   *   "pioneer": "not_legal",
   *   "explorer": "not_legal",
   *   "modern": "not_legal",
   *   "legacy": "legal",
   *   "pauper": "not_legal",
   *   "vintage": "legal",
   *   "penny": "not_legal",
   *   "commander": "legal",
   *   "oathbreaker": "legal",
   *   "standardbrawl": "not_legal",
   *   "brawl": "not_legal",
   *   "alchemy": "not_legal",
   *   "paupercommander": "not_legal",
   *   "duel": "legal",
   *   "oldschool": "not_legal",
   *   "premodern": "not_legal",
   *   "predh": "not_legal"
   * },
   * "games": [
   *   "paper",
   *   "mtgo"
   * ],
   * "reserved": false,
   * "game_changer": false,
   * "foil": false,
   * "nonfoil": true,
   * "finishes": [
   *   "nonfoil"
   * ],
   * "oversized": false,
   * "promo": false,
   * "reprint": false,
   * "variation": false,
   * "set_id": "f35f5dd8-5b95-4f52-bbe7-1c62909a8d08",
   * "set": "40k",
   * "set_name": "Warhammer 40,000 Commander",
   * "set_type": "commander",
   * "set_uri": "https://api.scryfall.com/sets/f35f5dd8-5b95-4f52-bbe7-1c62909a8d08",
   * "set_search_uri": "https://api.scryfall.com/cards/search?order=set&q=e%3A40k&unique=prints",
   * "scryfall_set_uri": "https://scryfall.com/sets/40k?utm_source=api",
   * "rulings_uri": "https://api.scryfall.com/cards/b8c69031-588d-44a2-89b3-972a335541e5/rulings",
   * "prints_search_uri": "https://api.scryfall.com/cards/search?order=released&q=oracleid%3Af96fcb37-5f17-4231-b1a2-64f5fe45730b&unique=prints",
   * "collector_number": "57",
   * "digital": false,
   * "rarity": "uncommon",
   * "flavor_text": "Nothing can override the hard-wired desire to kill that empowers these rage-driven Necrons.",
   * "card_back_id": "0aeebaf5-8c7d-4636-9e82-8c27447861f7",
   * "artist": "Games Workshop",
   * "artist_ids": [
   *   "9456da4e-410b-4932-8bf3-a69a2da3cdc9"
   * ],
   * "illustration_id": "210df945-d937-434c-8959-1ae1b39bb205",
   * "border_color": "black",
   * "frame": "2015",
   * "security_stamp": "triangle",
   * "full_art": false,
   * "textless": false,
   * "booster": false,
   * "story_spotlight": false,
   * "edhrec_rank": 13568,
   * "preview": {
   *   "source": "The Command Zone",
   *   "source_uri": "https://www.youtube.com/watch?v=g2UWu8lKU5k",
   *   "previewed_at": "2022-09-15"
   * },
   * "prices": {
   *   "usd": "0.20",
   *   "usd_foil": null,
   *   "usd_etched": null,
   *   "eur": "0.18",
   *   "eur_foil": null,
   *   "tix": "0.50"
   * },
   * "related_uris": {
   *   "gatherer": "https://gatherer.wizards.com/Pages/Card/Details.aspx?multiverseid=580879&printed=false",
   *   "tcgplayer_infinite_articles": "https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&trafcat=infinite&u=https%3A%2F%2Finfinite.tcgplayer.com%2Fsearch%3FcontentMode%3Darticle%26game%3Dmagic%26q%3DSkorpekh%2BDestroyer",
   *   "tcgplayer_infinite_decks": "https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&trafcat=infinite&u=https%3A%2F%2Finfinite.tcgplayer.com%2Fsearch%3FcontentMode%3Ddeck%26game%3Dmagic%26q%3DSkorpekh%2BDestroyer",
   *   "edhrec": "https://edhrec.com/route/?cc=Skorpekh+Destroyer"
   * },
   * "purchase_uris": {
   *   "tcgplayer": "https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&u=https%3A%2F%2Fwww.tcgplayer.com%2Fproduct%2F286354%3Fpage%3D1",
   *   "cardmarket": "https://www.cardmarket.com/en/Magic/Products/Singles/Universes-Beyond-Warhammer-40000/Skorpekh-Destroyer?referrer=scryfall&utm_campaign=card_prices&utm_medium=text&utm_source=scryfall",
   *   "cardhoarder": "https://www.cardhoarder.com/cards/108586?affiliate_id=scryfall&ref=card-profile&utm_campaign=affiliate&utm_medium=card&utm_source=scryfall"
   * }
   */
  object?: string;
  id?: string;
  oracle_id?: string;
  multiverse_ids?: number[];
  mtgo_id?: number;
  tcgplayer_id?: number;
  cardmarket_id?: number;
  name?: string;
  lang?: string;
  released_at?: string;
  uri?: string;
  scryfall_uri?: string;
  layout?: string;
  highres_image?: boolean;
  image_status?: string;
  image_uris?: {
    small: string;
    normal: string;
    large: string;
    png: string;
    art_crop: string;
    border_crop: string;
  };
  mana_cost?: string;
  cmc?: number;
  type_line?: string;
  oracle_text?: string;
  power?: string;
  toughness?: string;
  colors?: string[];
  color_identity?: string[];
  keywords?: string[];
  legalities?: {
    standard: string;
    future: string;
    historic: string;
    timeless: string;
    gladiator: string;
    pioneer: string;
    explorer: string;
    modern: string;
    legacy: string;
    pauper: string;
    vintage: string;
    penny: string;
    commander: string;
    oathbreaker: string;
    standardbrawl: string;
    brawl: string;
    alchemy: string;
    paupercommander: string;
    duel: string;
    oldschool: string;
    premodern: string;
  };
  games?: string[];
  reserved?: boolean;
  game_changer?: boolean;
  foil?: boolean;
  nonfoil?: boolean;
  finishes?: string[];
  oversized?: boolean;
  promo?: boolean;
  reprint?: boolean;
  variation?: boolean;
  set_id?: string;
  set?: string;
  set_name?: string;
  set_type?: string;
  set_uri?: string;
  set_search_uri?: string;
  scryfall_set_uri?: string;
  rulings_uri?: string;
  prints_search_uri?: string;
  collector_number?: string;
  digital?: boolean;
  rarity?: string;
  flavor_text?: string;
  card_back_id?: string;
  artist?: string;
  artist_ids?: string[];
  illustration_id?: string;
  border_color?: string;
  frame?: string;
  security_stamp?: string;
  full_art?: boolean;
  textless?: boolean;
  booster?: boolean;
  story_spotlight?: boolean;
  edhrec_rank?: number;
  preview?: {
    source: string;
    source_uri: string;
    previewed_at: string;
  };
  prices?: {
    usd: string;
    usd_foil: string | null;
    usd_etched: string | null;
    eur: string;
    eur_foil: string | null;
    tix: string;
  };
  related_uris?: {
    gatherer: string;
    tcgplayer_infinite_articles: string;
    tcgplayer_infinite_decks: string;
    edhrec: string;
  };
  purchase_uris?: {
    tcgplayer: string;
    cardmarket: string;
    cardhoarder: string;
  };
}

export interface Match {
  id: string;
  name: string;
  score: number;

  set_name?: string;
  set_code?: string;
  img_uri?: string;

  all_data?: ScryfallCardData;

  // computed data on received
  extra_data?: {
    mana_cost: (string | TemplateResult)[];
    oracle_text: (string | TemplateResult)[];
  };
}

export interface Detection {
  id: number;
  color: string;
  points: number[][];
  polygon: number[][];
  polygon_closed: number[][];
  img: string;
  matches: Match[];
}

export interface Stats {
  messagesSent: number;
  messagesReceived: number;
  // server response data
  serverProcessTime: number | null;
  serverProcessPeriod: number | null;
  serverRecvImBytes: number | null;
  serverSendImBytes: number | null;
}

export interface Payload {
  detections: Detection[];
  server_process_time: number;
  server_process_period: number;
  server_recv_im_bytes: number;
  server_send_im_bytes: number;
}

export type SvgInHtml = HTMLElement & SVGElement;

export interface ScryfallCardSymbol {
  /**
   * From https://api.scryfall.com/symbology
   * e.g.
   * {"object":"card_symbol","symbol":"{T}","svg_uri":"https://svgs.scryfall.io/card-symbols/T.svg","loose_variant":null,"english":"tap this permanent","transposable":false,"represents_mana":false,"appears_in_mana_costs":false,"mana_value":0.0,"hybrid":false,"phyrexian":false,"cmc":0.0,"funny":false,"colors":[],"gatherer_alternates":["ocT","oT"]},
   * {"object":"card_symbol","symbol":"{Q}","svg_uri":"https://svgs.scryfall.io/card-symbols/Q.svg","loose_variant":null,"english":"untap this permanent","transposable":false,"represents_mana":false,"appears_in_mana_costs":false,"mana_value":0.0,"hybrid":false,"phyrexian":false,"cmc":0.0,"funny":false,"colors":[],"gatherer_alternates":null},
   */
  object: "card_symbol";
  symbol: string;
  svg_uri: string;
  loose_variant: string | null;
  english: string;
  transposable: boolean;
  represents_mana: boolean;
  appears_in_mana_costs: boolean;
  mana_value: number;
  hybrid: boolean;
  phyrexian: boolean;
  cmc: number;
  funny: boolean;
  colors: string[];
  gatherer_alternates: string[] | null;
}

export interface ScryfallSymbolsResponse {
  object: "list";
  has_more: boolean;
  data: ScryfallCardSymbol[];
}
