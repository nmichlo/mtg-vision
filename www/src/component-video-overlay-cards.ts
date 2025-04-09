import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $detections, $selectedId } from './util-store';
import * as SVG from "svg.js";
import { Detection, SvgInHtml } from "./types";






/**
 * Represents a single card detection with its own SVG elements.
 */
class SvgCard {

  id: number;
  svg: SVG.Container;
  group: SVG.G;
  polygon: SVG.Polygon;
  textGroup: SVG.G;
  text: SVG.Text;
  onClick: (id: number) => void;

  constructor(detection: Detection, svg: SVG.Container, onClick: (id: number) => void) {
    this.id = detection.id;
    this.svg = svg;
    this.onClick = onClick;

    // Create SVG group and elements
    this.group = this.svg.group();
    this.polygon = this.group.polygon([])
      .fill('rgba(0, 255, 0, 0.0)')
      .stroke({ color: detection.color, width: 2 })
      .attr('pointer-events', 'all');
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
    // draw polygon
    const pointsStr = detection.points.map(p => p.join(',')).join(' ');
    this.polygon.plot(pointsStr);
    this.polygon.stroke({ color: isSelected ? 'yellow' : detection.color, width: isSelected ? 4 : 2 });

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

  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);

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
    this.drawDetections();
  }

  setDimensions(width: number, height: number) {
    this.originalWidth = width;
    this.originalHeight = height;
    this.updateOverlaySize();
  }

  drawDetections() {
    if (!this.originalWidth || !this.originalHeight) return;

    const currentIds = new Set();

    this.#detectionsController.value.forEach(det => {
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

    this.cardMap.forEach((card, id) => {
      if (!currentIds.has(id)) {
        card.remove();
        this.cardMap.delete(id);
      }
    });
  }

  updateOverlaySize = () => {
    if (this.originalWidth && this.originalHeight) {
      this.svg.viewbox(0, 0, this.originalWidth, this.originalHeight);
    }
  };
}
customElements.define('cards-overlay', CardsOverlay);
