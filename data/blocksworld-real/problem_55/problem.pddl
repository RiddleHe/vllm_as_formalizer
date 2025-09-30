(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        blue - block
        green - block
        yellow - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable blue)
        (clear blue)
        (ontable green)
        (clear green)
        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable orange)
        (clear orange)
        ; purple
        ; blue
        ; green
        ; yellow
        ; red
        ; orange
    )
    (:goal (and (on orange red)
        (on green orange)
        (on purple yellow)
        (on blue purple)))
)
    