import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class StateService {
  getD;
  clicked;
  resultReady;
  overFire;
  overNormal;
  isFireMode=false;
  isGreenMode=false;
  isNormalMode=true;
  constructor() { }
  setToFireState(){
    this.isFireMode=true;
    this.isGreenMode=false;
    this.isNormalMode=false;
  }
  setToGreenState(){
    this.isFireMode=false;
    this.isGreenMode=true
    this.isNormalMode=false;
  }
  setToNormalState(){
    this.isFireMode=false;
    this.isGreenMode=false;
    this.isNormalMode=true;
  }
}
