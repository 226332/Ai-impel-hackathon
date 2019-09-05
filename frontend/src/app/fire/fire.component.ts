import { Component, OnInit } from '@angular/core';
import {StateService} from '../state.service';

@Component({
  selector: 'app-fire',
  templateUrl: './fire.component.html',
  styleUrls: ['./fire.component.css']
})
export class FireComponent implements OnInit {

  constructor(public state:StateService) { }

  ngOnInit() {
  }

}
