import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import {
  $detections,
  $selectedId,
  $videoDimensions,
  $showOverlayXyxyxyxy,
  $showOverlayPolygonClosed,
  $showOverlayPolygon,
  getDetections, $matchThreshold
} from './util-store';
import * as SVG from "svg.js";
import { Detection, SvgInHtml } from "./types";






/**
 * Represents a single card detection with its own SVG elements.
 */
class SvgCard {

  id: number;
  svg: SVG.Container;
  group: SVG.G;

  points: SVG.Polygon;
  polygon: SVG.Polygon;
  polygonClosed: SVG.Polygon;

  textGroup: SVG.G;
  text: SVG.Text;
  onClick: (id: number) => void;

  constructor(detection: Detection, svg: SVG.Container, onClick: (id: number) => void) {
    this.id = detection.id;
    this.svg = svg;
    this.onClick = onClick;

    // Create SVG group and elements
    this.group = this.svg.group();
    this.points = this.group.polygon([])
      .fill('rgba(0, 255, 0, 0.0)')
      .stroke({ color: detection.color, width: 1 })
      .attr('pointer-events', 'all');
    this.polygon = this.group.polygon([]).stroke({ color: '#ffffff', width: 1 }).fill('rgba(255, 255, 255, 0.2)');
    this.polygonClosed = this.group.polygon([]).stroke({ color: '#000000', width: 1 }).fill('rgba(0, 0, 0, 0.1)');
    this.textGroup = this.group.group();  // translate this
    this.text = this.textGroup.text('')  // rotate this
      .font({ size: 10, style: 'fill: white', family: 'goudy, serif' })

    // Attach click handler to the group
    this.group.on('click', (e) => {
      e.stopPropagation(); // Prevent bubbling to SVG background
      this.onClick(this.id);
    });
  }

  update(detection: Detection, isSelected: boolean) {
    // draw xyxyxyxy
    const pointsStr = detection.points.map(p => p.join(',')).join(' ');
    this.points.plot(pointsStr);
    if ($showOverlayXyxyxyxy.get()) {
      this.points.stroke({color: isSelected ? 'yellow' : detection.color, width: isSelected ? 2 : 1});
    } else {
      this.points.stroke({color: 'rgba(0, 0, 0, 0.0)', width: 1});
    }

    // draw polygon
    if ($showOverlayPolygon.get()) {
      this.polygon.plot(detection.polygon.map(p => p.join(',')).join(' '));
    } else {
      this.polygon.plot([]);
    }

    // draw polygon closed
    if ($showOverlayPolygonClosed.get()) {
      this.polygonClosed.plot(detection.polygon_closed.map(p => p.join(',')).join(' '));
    } else {
      this.polygonClosed.plot([]);
    }

    // draw text
    const bestMatch = detection.matches[0];
    if (bestMatch) {
      const [[x0, y0], [x1, y1]] = detection.points.slice(0, 2);
      const angle = Math.atan2(y1 - y0, x1 - x0) * (180 / Math.PI);
      this.text.text(bestMatch.name)
      this.text.transform({rotation: angle, cx: 0, cy: 0});
      this.textGroup.transform({x: x0, y: y0})
    }
  }

  /**
   * Removes the card’s SVG elements from the container.
   */
  remove() {
    this.group.remove();
  }
}




class CardsOverlay extends LitElement {

  // don't remove even though unused
  #thresholdController = new StoreController(this, $matchThreshold);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);
  #videoDimensionsController = new StoreController(this, $videoDimensions);

  static styles = css`
    :host {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: auto;
      z-index: 1;
    }
    svg {
      width: 100%;
      height: 100%;
      pointer-events: auto;
    }
  `;

  cardMap: Map<number, SvgCard>;
  svgElement: SvgInHtml;
  svg: SVG.Container;
  originalWidth: number | null = null;
  originalHeight: number | null = null;

  constructor() {
    super();
    this.cardMap = new Map(); // Store Card instances by detection ID
  }

  render() {
    return html`
      <svg id="overlay"></svg>
    `;
  }

  firstUpdated() {
    this.svgElement = this.shadowRoot.getElementById('overlay') as SvgInHtml;
    this.svg = SVG(this.svgElement);

    // Add click handler to SVG background to clear selection
    this.svg.on('click', (e) => {
      if (e.target === this.svgElement) { // Check if click is on the SVG itself, not a child
        $selectedId.set(null);
      }
    });

    // Add window resize listener
    window.addEventListener('resize', this.updateOverlaySize);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('resize', this.updateOverlaySize);
  }

  updated() {
    // Check if video dimensions are available
    const dimensions = this.#videoDimensionsController.value;
    if (dimensions) {
      this.originalWidth = dimensions.width;
      this.originalHeight = dimensions.height;
      this.updateOverlaySize();
    }

    this.drawDetections();
  }

  drawDetections() {
    if (!this.originalWidth || !this.originalHeight) return;

    const currentIds = new Set();
    const detections = getDetections(); // this.#detectionsController.value;

    // Auto-select the first card if there are detections and nothing is currently selected
    if (detections.length > 0 && this.#selectedIdController.value === null) {
      $selectedId.set(detections[0].id);
    }

    detections.forEach(det => {
      const id = det.id;
      currentIds.add(id);
      let card = this.cardMap.get(id);

      if (!card) {
        card = new SvgCard(det, this.svg, (clickedId) => {
          // Only select if not already selected
          if (this.#selectedIdController.value !== clickedId) {
            $selectedId.set(clickedId);
          }
          // Do nothing if already selected, keeping it clicked
        });
        this.cardMap.set(id, card);
      }

      const isSelected = id === this.#selectedIdController.value;
      card.update(det, isSelected);
    });

    // Handle removed cards
    const selectedId = this.#selectedIdController.value;
    let selectedCardRemoved = false;
    this.cardMap.forEach((card, id) => {
      if (!currentIds.has(id)) {
        if (id === selectedId) {
          selectedCardRemoved = true;
        }
        card.remove();
        this.cardMap.delete(id);
      }
    });

    // If the selected card was removed
    // - Select the first available one if we have other cards
    // - Clear selection if no cards are left
    if (selectedCardRemoved) {
      if (detections.length > 0) {
        $selectedId.set(detections[0].id);
      } else {
        $selectedId.set(null);
      }
    }
  }

  updateOverlaySize = () => {
    if (this.originalWidth && this.originalHeight) {
      this.svg.viewbox(0, 0, this.originalWidth, this.originalHeight);
    }
  };
}
customElements.define('cards-overlay', CardsOverlay);
