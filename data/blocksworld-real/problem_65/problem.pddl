(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        orange - block
        green - block
        blue - block
        purple - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable orange)
        (clear orange)
        (ontable green)
        (clear green)
        (ontable blue)
        (on purple blue)
        (on red purple)
        (clear red)
        ; yellow
        ; orange
        ; green
        ; blue -> purple -> red
    )
    (:goal (and (on green red)
        (on blue orange)
        (on purple blue)
        (on yellow purple)))
)
    