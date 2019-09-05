import {Component, OnInit, ViewChild} from '@angular/core';
import {StateService} from '../state.service';
import {animate, state, style, transition, trigger} from '@angular/animations';
declare var particlesJS: any;
@Component({
  selector: 'app-background',
  templateUrl: './background.component.html',
  styleUrls: ['./background.component.css'],
  animations: [
    trigger(
      'inOutAnimation',
      [
        transition(
          ':enter',
          [
            style({  opacity: 0 }),
            animate('2s ease-out',
              style({ opacity: 1 }))
          ]
        ),
        transition(
          ':leave',
          [
            style({ opacity: 1 }),
            animate('1s ease-in',
              style({  opacity: 0 }))
          ]
        )
      ]
    )
  ]
})
export class BackgroundComponent implements OnInit {

  get d(){
    return this.state.getD();
  }
  constructor(public state:StateService) { }

  ngOnInit() {
    particlesJS.load('particles-js', 'assets/particlesjs-config.json', () => {
      console.log('callback - particles.js config loaded');
    });

  }

}
