/**
 * Card Preview Component
 *
 * Defines a custom web component for displaying card preview information in the
 * MTG Vision application. This component renders:
 * - A captured image of the detected card
 * - The tracking ID and card name (if identified)
 * - A small preview of the matched card from Scryfall (if available)
 *
 * Used in the sidebar to display detected cards from the video stream.
 */

import { css, html, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";
import { ObjectEmbeddingInfo } from "./types";

/**
 * Card preview component for displaying detected card information
 */
@customElement("card-preview")
export class CardPreview extends LitElement {
  static styles = css`
    :host {
      margin-bottom: 10px;
      background-color: #333;
      border-radius: 4px;
      overflow: hidden;
      position: relative;
      display: block;
    }
    .preview-image {
      width: 100%;
      display: block;
    }
    .info {
      padding: 5px;
      color: white;
      font-size: 0.8em;
      display: flex;
      justify-content: space-between;
    }
    .card-pip {
      position: absolute;
      bottom: 0;
      right: 0;
      width: 40%;
      height: 40%;
      background-color: rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .card-pip img {
      max-width: 90%;
      max-height: 90%;
      object-fit: contain;
    }
  `;

  @property({ type: Number }) objectId!: number;
  @property({ type: String }) image!: string;
  @property({ type: Object }) cardInfo?: ObjectEmbeddingInfo;

  render() {
    return html`
      <img class="preview-image" src="${this.image}" alt="Card preview ${this.objectId}">
      <div class="info">
        <span>ID: ${this.objectId}</span>
        ${this.cardInfo?.cardData ? html`
          <span>${this.cardInfo.cardData.name}</span>
        ` : ''}
      </div>
      ${this.cardInfo?.cardData?.image_uris?.small ? html`
        <div class="card-pip">
          <img src="${this.cardInfo.cardData.image_uris.small}"
               alt="${this.cardInfo.cardData.name}">
        </div>
      ` : ''}
    `;
  }
}
